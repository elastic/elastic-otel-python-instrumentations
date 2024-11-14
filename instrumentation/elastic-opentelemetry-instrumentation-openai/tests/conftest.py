# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright
# ownership. Elasticsearch B.V. licenses this file to you under
# the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
from typing import Sequence, Union
from urllib.parse import parse_qs, urlparse

import openai
import pytest
from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry._events import set_event_logger_provider
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.metrics import Histogram
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test.globals_test import reset_metrics_globals

from .utils import assert_metric_expected, create_histogram_data_point


@pytest.fixture(scope="session")
def trace_exporter():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)

    provider = TracerProvider()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return exporter


# cannot use a fixture session scoped because we don't have a way to clear it
# TODO: expose api to reset it upstream?
@pytest.fixture
def metrics_reader():
    reset_metrics_globals()
    memory_reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[memory_reader])
    metrics.set_meter_provider(provider)

    return memory_reader


@pytest.fixture(scope="session")
def logs_exporter():
    exporter = InMemoryLogExporter()
    logger_provider = LoggerProvider()
    set_logger_provider(logger_provider)

    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))

    event_logger_provider = EventLoggerProvider(logger_provider=logger_provider)
    set_event_logger_provider(event_logger_provider)

    return exporter


@pytest.fixture(autouse=True)
def clear_exporter(trace_exporter, metrics_reader, logs_exporter):
    trace_exporter.clear()
    logs_exporter.clear()


# TODO: should drop autouse and use it explicitly?
@pytest.fixture(autouse=True)
def instrument():
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument()

    yield instrumentor

    instrumentor.uninstrument()


OPENAI_API_KEY = "test_openai_api_key"
OPENAI_ORG_ID = "test_openai_org_key"
OPENAI_PROJECT_ID = "test_openai_project_id"


@pytest.fixture
def default_openai_env(monkeypatch):
    """
    This fixture prevents OpenAIProvider.from_env() from erring on missing
    environment variables.

    When running VCR tests for the first time or after deleting a cassette
    recording, set required environment variables, so that real requests don't
    fail. Subsequent runs use the recorded data, so don't need them.
    """
    if "OPENAI_API_KEY" not in os.environ:
        monkeypatch.setenv("OPENAI_API_KEY", OPENAI_API_KEY)


AZURE_ENDPOINT = "https://test.openai.azure.com"
AZURE_DEPLOYMENT_NAME = "test-azure-deployment"
AZURE_API_KEY = "test_azure_api_key"
AZURE_CHAT_COMPLETIONS_API_VERSION = "2023-03-15-preview"
AZURE_CHAT_COMPLETIONS_DEPLOYMENT_URL = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_CHAT_COMPLETIONS_API_VERSION}"
AZURE_EMBEDDINGS_API_VERSION = "2023-05-15"
AZURE_EMBEDDINGS_DEPLOYMENT_URL = (
    f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT_NAME}/embeddings?api-version={AZURE_EMBEDDINGS_API_VERSION}"
)


@pytest.fixture
def default_azure_env(monkeypatch):
    """
    This fixture prevents AzureProvider.from_env() from erring on missing
    environment variables.

    When running VCR tests for the first time or after deleting a cassette
    recording, set required environment variables, so that real requests don't
    fail. Subsequent runs use the recorded data, so don't need them.
    """
    if "AZURE_CHAT_COMPLETIONS_DEPLOYMENT_URL" not in os.environ:
        monkeypatch.setenv("AZURE_CHAT_COMPLETIONS_DEPLOYMENT_URL", AZURE_CHAT_COMPLETIONS_DEPLOYMENT_URL)
    if "AZURE_CHAT_COMPLETIONS_API_KEY" not in os.environ:
        monkeypatch.setenv("AZURE_CHAT_COMPLETIONS_API_KEY", AZURE_API_KEY)
    if "AZURE_EMBEDDINGS_DEPLOYMENT_URL" not in os.environ:
        monkeypatch.setenv("AZURE_EMBEDDINGS_DEPLOYMENT_URL", AZURE_EMBEDDINGS_DEPLOYMENT_URL)
    if "AZURE_EMBEDDINGS_API_KEY" not in os.environ:
        monkeypatch.setenv("AZURE_EMBEDDINGS_API_KEY", AZURE_API_KEY)


@pytest.fixture(scope="module")
def vcr_config():
    """
    This scrubs sensitive data and gunzips bodies when in recording mode.

    Without this, you would leak cookies and auth tokens in the cassettes.
    Also, depending on the request, some responses would be binary encoded
    while others plain json. This ensures all bodies are human-readable.
    """
    return {
        "decode_compressed_response": True,
        "filter_headers": [
            ("authorization", "Bearer " + OPENAI_API_KEY),
            ("openai-organization", OPENAI_ORG_ID),
            ("openai-project", OPENAI_PROJECT_ID),
            ("cookie", None),
        ],
        "before_record_request": scrub_request_url,
        "before_record_response": scrub_response_headers,
    }


def scrub_request_url(request):
    """
    This scrubs sensitive request data in provider-specific way. Note that headers
    are case-sensitive!
    """
    if "openai.azure.com" in request.uri:
        request.uri = re.sub(r"https://[^/]+", AZURE_ENDPOINT, request.uri)
        request.uri = re.sub(r"/deployments/[^/]+", f"/deployments/{AZURE_DEPLOYMENT_NAME}", request.uri)
        request.headers["host"] = AZURE_ENDPOINT.replace("https://", "")
        request.headers["api-key"] = AZURE_API_KEY

    return request


def scrub_response_headers(response):
    """
    This scrubs sensitive response headers. Note they are case-sensitive!
    """
    response["headers"]["openai-organization"] = OPENAI_ORG_ID
    response["headers"]["Set-Cookie"] = "test_set_cookie"
    return response


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    # remove test_ prefix from dir name
    return os.path.join("tests", "cassettes", request.module.__name__[5:])


@pytest.fixture
def vcr_cassette_name(request):
    """Name of the VCR cassette"""
    test_class = request.cls
    if test_class:
        return "{}.{}".format(test_class.__name__, request.node.name)

    # We have chicken and egg problem when using the output of the call as fixture because llm responses are
    # not reproducible and so that name of the cassette would not match the content
    # For the convenience of being able to use the test data as parametrized fixture we assume that the first
    # element of the test data is the provider and remove everything else from the cassette name
    # TODO: we can probably rebuild the name from the request
    # request.node.name format: test_basic[ollama_provider_chat_completions-Atlantic Ocean.-24-4-5]
    test_name = re.match(r"(\w+\[\w+)", request.node.name)
    return f"{test_name.group()}]"


class AzureProvider:
    def __init__(self, api_key, endpoint, api_version, operation_name):
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version

        self.operation_name = operation_name

        self.server_address = "test.openai.azure.com"
        self.server_port = 443

    @classmethod
    def from_env(cls, operation_name):
        if operation_name == "chat":
            deployment_url = os.getenv("AZURE_CHAT_COMPLETIONS_DEPLOYMENT_URL")
            api_key = os.getenv("AZURE_CHAT_COMPLETIONS_API_KEY")
        elif operation_name == "embeddings":
            deployment_url = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT_URL")
            api_key = os.getenv("AZURE_EMBEDDINGS_API_KEY")
        else:
            raise NotImplementedError()

        parsed_url = urlparse(deployment_url)
        endpoint = f"https://{parsed_url.hostname}"
        parsed_qs = parse_qs(parsed_url.query)
        # query string entries are lists
        api_version = parsed_qs["api-version"][0]
        return cls(
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            operation_name=operation_name,
        )

    def _get_client_kwargs(self):
        return {
            "api_key": self.api_key,
            "azure_endpoint": self.endpoint,
            "api_version": self.api_version,
            "max_retries": 1,
        }

    def get_client(self):
        return openai.AzureOpenAI(**self._get_client_kwargs())

    def get_async_client(self):
        return openai.AsyncAzureOpenAI(**self._get_client_kwargs())


class OpenAIProvider:
    def __init__(self, api_key, operation_name):
        self.api_key = api_key

        self.operation_name = operation_name

        self.server_address = "api.openai.com"
        self.server_port = 443

    @classmethod
    def from_env(cls, operation_name):
        api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        return cls(api_key=api_key, operation_name=operation_name)

    def _get_client_kwargs(self):
        return {
            "api_key": self.api_key,
            "max_retries": 1,
        }

    def get_client(self):
        return openai.OpenAI(**self._get_client_kwargs())

    def get_async_client(self):
        return openai.AsyncOpenAI(**self._get_client_kwargs())


class OllamaProvider:
    def __init__(self, operation_name):
        self.api_key = "unused"

        self.operation_name = operation_name

        self.server_address = "localhost"
        self.server_port = 11434

    @classmethod
    def from_env(cls, operation_name):
        return cls(operation_name=operation_name)

    def _get_client_kwargs(self):
        return {
            "base_url": f"http://{self.server_address}:{self.server_port}/v1",
            "api_key": self.api_key,
            "max_retries": 1,
        }

    def get_client(self):
        return openai.OpenAI(**self._get_client_kwargs())

    def get_async_client(self):
        return openai.AsyncOpenAI(**self._get_client_kwargs())


@pytest.fixture
def azure_provider_chat_completions(default_azure_env):
    return AzureProvider.from_env(operation_name="chat")


@pytest.fixture
def openai_provider_chat_completions(default_openai_env):
    return OpenAIProvider.from_env(operation_name="chat")


@pytest.fixture
def ollama_provider_chat_completions():
    return OllamaProvider.from_env(operation_name="chat")


@pytest.fixture
def azure_provider_embeddings(default_azure_env):
    return AzureProvider.from_env(operation_name="embeddings")


@pytest.fixture
def openai_provider_embeddings(default_openai_env):
    return OpenAIProvider.from_env(operation_name="embeddings")


@pytest.fixture
def ollama_provider_embeddings():
    return OllamaProvider.from_env(operation_name="embeddings")


def assert_operation_duration_metric(
    provider,
    metric: Histogram,
    attributes: dict,
    min_data_point: float,
    max_data_point: float = None,
    sum_data_point: float = None,
    count: int = 1,
):
    assert metric.name == "gen_ai.client.operation.duration"
    default_attributes = {
        "gen_ai.operation.name": provider.operation_name,
        "gen_ai.system": "openai",
        "server.address": provider.server_address,
        "server.port": provider.server_port,
    }
    # handle the simple cases of 1 or 2 data points in the histogram with just min_data_point
    if max_data_point is None:
        max_data_point = min_data_point
    if sum_data_point is None:
        if count == 1:
            sum_data_point = min_data_point
        elif count == 2:
            sum_data_point = min_data_point + max_data_point
        else:
            raise ValueError("Missing sum_data_point with more than 2 values")
    assert_metric_expected(
        metric,
        [
            create_histogram_data_point(
                count=count,
                sum_data_point=sum_data_point,
                max_data_point=max_data_point,
                min_data_point=min_data_point,
                attributes={**default_attributes, **attributes},
            ),
        ],
        est_value_delta=0.25,
    )


def assert_error_operation_duration_metric(
    provider, metric: Histogram, attributes: dict, data_point: float, value_delta: float = 0.5
):
    assert metric.name == "gen_ai.client.operation.duration"
    default_attributes = {
        "gen_ai.operation.name": provider.operation_name,
        "gen_ai.system": "openai",
        "error.type": "APIConnectionError",
        "server.address": "localhost",
        "server.port": 9999,
    }
    assert_metric_expected(
        metric,
        [
            create_histogram_data_point(
                count=1,
                sum_data_point=data_point,
                max_data_point=data_point,
                min_data_point=data_point,
                attributes={**default_attributes, **attributes},
            ),
        ],
        est_value_delta=value_delta,
    )


def assert_token_usage_input_metric(
    provider, metric: Histogram, attributes: dict, input_data_point: int, count: int = 1
):
    assert metric.name == "gen_ai.client.token.usage"
    default_attributes = {
        "gen_ai.operation.name": provider.operation_name,
        "gen_ai.system": "openai",
        "server.address": provider.server_address,
        "server.port": provider.server_port,
        "gen_ai.token.type": "input",
    }
    assert_metric_expected(
        metric,
        [
            create_histogram_data_point(
                count=count,
                sum_data_point=input_data_point,
                max_data_point=input_data_point,
                min_data_point=input_data_point,
                attributes={**default_attributes, **attributes},
            ),
        ],
    )


def assert_token_usage_metric(
    provider,
    metric: Histogram,
    attributes: dict,
    input_data_point: Union[int, Sequence[int]],
    output_data_point: Union[int, Sequence[int]],
    count: int = 1,
):
    assert metric.name == "gen_ai.client.token.usage"
    default_attributes = {
        "gen_ai.operation.name": provider.operation_name,
        "gen_ai.system": "openai",
        "server.address": provider.server_address,
        "server.port": provider.server_port,
        "gen_ai.token.type": "input",
    }

    if count == 1:
        assert isinstance(input_data_point, int)
        assert isinstance(output_data_point, int)
        sum_input = min_input = max_input = input_data_point
        sum_output = min_output = max_output = output_data_point
    else:
        assert isinstance(input_data_point, Sequence)
        assert isinstance(output_data_point, Sequence)
        sum_input = sum(input_data_point)
        min_input = min(input_data_point)
        max_input = max(input_data_point)
        sum_output = sum(output_data_point)
        min_output = min(output_data_point)
        max_output = max(output_data_point)

    assert_metric_expected(
        metric,
        [
            create_histogram_data_point(
                count=count,
                sum_data_point=sum_input,
                max_data_point=max_input,
                min_data_point=min_input,
                attributes={**default_attributes, **attributes, "gen_ai.token.type": "input"},
            ),
            create_histogram_data_point(
                count=count,
                sum_data_point=sum_output,
                max_data_point=max_output,
                min_data_point=min_output,
                attributes={**default_attributes, **attributes, "gen_ai.token.type": "output"},
            ),
        ],
    )
