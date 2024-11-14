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

import openai
import pytest
from opentelemetry.instrumentation.openai.helpers import GEN_AI_REQUEST_ENCODING_FORMAT
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS, SERVER_PORT

from .conftest import (
    assert_error_operation_duration_metric,
    assert_operation_duration_metric,
    assert_token_usage_input_metric,
)
from .utils import get_sorted_metrics


test_basic_test_data = [
    ("openai_provider_embeddings", "text-embedding-3-small", 4, 0.2263190783560276),
    ("azure_provider_embeddings", "text-embedding-3-small", 4, 0.0017870571464300156),
    ("ollama_provider_embeddings", "all-minilm:33m", 4, 0.0030461717396974564),
]


@pytest.mark.vcr()
@pytest.mark.parametrize("provider_str,model,input_tokens,duration", test_basic_test_data)
def test_basic(provider_str, model, input_tokens, duration, trace_exporter, metrics_reader, request):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    text = "South Atlantic Ocean."
    response = client.embeddings.create(model=model, input=[text])

    assert len(response.data) == 1

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: provider.operation_name,
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_MODEL: model,
        GEN_AI_USAGE_INPUT_TOKENS: 4,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert span.events == ()

    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: model,
    }
    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_input_metric(provider, token_usage_metric, attributes=attributes, input_data_point=input_tokens)


test_all_the_client_options_test_data = [
    ("openai_provider_embeddings", "text-embedding-3-small", 4, 0.2263190783560276),
    ("azure_provider_embeddings", "text-embedding-3-small", 4, 0.0017870571464300156),
    ("ollama_provider_embeddings", "all-minilm:33m", 4, 0.0030461717396974564),
]


@pytest.mark.vcr()
@pytest.mark.parametrize("provider_str,model,input_tokens,duration", test_all_the_client_options_test_data)
def test_all_the_client_options(provider_str, model, input_tokens, duration, trace_exporter, metrics_reader, request):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    text = "South Atlantic Ocean."
    response = client.embeddings.create(model=model, input=[text], encoding_format="float")

    assert len(response.data) == 1

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: provider.operation_name,
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_MODEL: model,
        GEN_AI_REQUEST_ENCODING_FORMAT: "float",
        GEN_AI_USAGE_INPUT_TOKENS: 4,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, data_point=0.2263190783560276
    )
    assert_token_usage_input_metric(provider, token_usage_metric, attributes=attributes, input_data_point=4)


test_connection_error_data = [
    ("openai_provider_embeddings", "text-embedding-3-small", 0.2263190783560276),
    ("azure_provider_embeddings", "text-embedding-3-small", 0.0017870571464300156),
    ("ollama_provider_embeddings", "all-minilm:33m", 0.0030461717396974564),
]


@pytest.mark.vcr()
@pytest.mark.parametrize("provider_str,model,duration", test_connection_error_data)
def test_connection_error(provider_str, model, duration, trace_exporter, metrics_reader, request):
    provider = request.getfixturevalue(provider_str)

    client = openai.Client(base_url="http://localhost:9999/v5", api_key="unused", max_retries=1)
    text = "South Atlantic Ocean."

    with pytest.raises(Exception):
        client.embeddings.create(model=model, input=[text])

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: provider.operation_name,
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        ERROR_TYPE: "APIConnectionError",
        SERVER_ADDRESS: "localhost",
        SERVER_PORT: 9999,
    }
    assert span.events == ()

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        ERROR_TYPE: "APIConnectionError",
    }
    assert_error_operation_duration_metric(
        provider,
        operation_duration_metric,
        attributes=attributes,
        data_point=duration,
        value_delta=1.0,
    )


test_async_basic_test_data = [
    ("openai_provider_embeddings", "text-embedding-3-small", 4, 0.2263190783560276),
    ("azure_provider_embeddings", "text-embedding-3-small", 4, 0.0017870571464300156),
    ("ollama_provider_embeddings", "all-minilm:33m", 4, 0.0030461717396974564),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize("provider_str,model,input_tokens,duration", test_async_basic_test_data)
async def test_async_basic(provider_str, model, input_tokens, duration, trace_exporter, metrics_reader, request):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_async_client()

    text = "South Atlantic Ocean."
    response = await client.embeddings.create(model=model, input=[text])

    assert len(response.data) == 1

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: provider.operation_name,
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_MODEL: model,
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_input_metric(provider, token_usage_metric, attributes=attributes, input_data_point=input_tokens)


test_async_all_the_client_options_test_data = [
    ("openai_provider_embeddings", "text-embedding-3-small", 4, 0.2263190783560276),
    ("azure_provider_embeddings", "text-embedding-3-small", 4, 0.0017870571464300156),
    ("ollama_provider_embeddings", "all-minilm:33m", 4, 0.0030461717396974564),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize("provider_str,model,input_tokens,duration", test_async_all_the_client_options_test_data)
async def test_async_all_the_client_options(
    provider_str, model, input_tokens, duration, trace_exporter, metrics_reader, request
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_async_client()

    text = "South Atlantic Ocean."
    response = await client.embeddings.create(model=model, input=[text], encoding_format="float")

    assert len(response.data) == 1

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: provider.operation_name,
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_MODEL: model,
        GEN_AI_REQUEST_ENCODING_FORMAT: "float",
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_input_metric(provider, token_usage_metric, attributes=attributes, input_data_point=input_tokens)


test_async_connection_error_test_data = [
    ("openai_provider_embeddings", "text-embedding-3-small", 0.2263190783560276),
    ("azure_provider_embeddings", "text-embedding-3-small", 0.0036478489999751673),
    ("ollama_provider_embeddings", "all-minilm:33m", 0.0030461717396974564),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize("provider_str,model,duration", test_async_connection_error_test_data)
async def test_async_connection_error(provider_str, model, duration, trace_exporter, metrics_reader, request):
    provider = request.getfixturevalue(provider_str)

    client = openai.AsyncOpenAI(base_url="http://localhost:9999/v5", api_key="unused", max_retries=1)
    text = "South Atlantic Ocean."

    with pytest.raises(Exception):
        await client.embeddings.create(model=model, input=[text])

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: provider.operation_name,
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        ERROR_TYPE: "APIConnectionError",
        SERVER_ADDRESS: "localhost",
        SERVER_PORT: 9999,
    }

    assert span.events == ()

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        ERROR_TYPE: "APIConnectionError",
    }
    assert_error_operation_duration_metric(
        provider,
        operation_duration_metric,
        attributes=attributes,
        data_point=duration,
        value_delta=1.0,
    )


test_without_model_parameter_test_data = [
    ("openai_provider_embeddings", "api.openai.com", 443, 4.2263190783560276),
    ("azure_provider_embeddings", "test.openai.azure.com", 443, 4.0017870571464300156),
    ("ollama_provider_embeddings", "localhost", 11434, 4.10461717396974564),
]


@pytest.mark.vcr()
@pytest.mark.parametrize("provider_str,server_address,server_port,duration", test_without_model_parameter_test_data)
def test_without_model_parameter(
    provider_str, server_address, server_port, duration, trace_exporter, metrics_reader, request
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    text = "South Atlantic Ocean."
    with pytest.raises(TypeError, match=re.escape("create() missing 1 required keyword-only argument: 'model'")):
        client.embeddings.create(input=[text])

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "embeddings"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    assert dict(span.attributes) == {
        ERROR_TYPE: "TypeError",
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_SYSTEM: "openai",
        SERVER_ADDRESS: server_address,
        SERVER_PORT: server_port,
    }

    attributes = {
        "error.type": "TypeError",
        "server.address": server_address,
        "server.port": server_port,
        "gen_ai.operation.name": "embeddings",
    }
    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    assert_error_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, data_point=duration, value_delta=5
    )


test_model_not_found_test_data = [
    (
        "openai_provider_embeddings",
        "api.openai.com",
        443,
        openai.NotFoundError,
        "The model `not-found-model` does not exist or you do not have access to it.",
        0.05915193818509579,
    ),
    (
        "azure_provider_embeddings",
        "test.openai.azure.com",
        443,
        openai.NotFoundError,
        "The API deployment for this resource does not exist",
        0.0015850383788347244,
    ),
    (
        "ollama_provider_embeddings",
        "localhost",
        11434,
        openai.NotFoundError,
        'model "not-found-model" not found, try pulling it first',
        0.087132233195006854,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,server_address,server_port,exception,exception_message,duration", test_model_not_found_test_data
)
def test_model_not_found(
    provider_str,
    server_address,
    server_port,
    exception,
    exception_message,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    text = "South Atlantic Ocean."
    with pytest.raises(exception, match=re.escape(exception_message)):
        client.embeddings.create(model="not-found-model", input=[text])

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "embeddings not-found-model"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    assert dict(span.attributes) == {
        ERROR_TYPE: exception.__qualname__,
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_REQUEST_MODEL: "not-found-model",
        GEN_AI_SYSTEM: "openai",
        SERVER_ADDRESS: server_address,
        SERVER_PORT: server_port,
    }

    attributes = {
        "error.type": exception.__qualname__,
        "server.address": server_address,
        "server.port": server_port,
        "gen_ai.operation.name": "embeddings",
        "gen_ai.request.model": "not-found-model",
    }
    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    assert_error_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, data_point=duration
    )
