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
import os
import re

import openai
import pytest
from opentelemetry.instrumentation.openai.helpers import GEN_AI_REQUEST_ENCODING_FORMATS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS, SERVER_PORT
from opentelemetry.trace import SpanKind, StatusCode

from .conftest import (
    address_and_port,
    assert_error_operation_duration_metric,
    assert_operation_duration_metric,
    assert_token_usage_input_metric,
    get_integration_async_client,
    get_integration_client,
)
from .utils import MOCK_POSITIVE_FLOAT, get_sorted_metrics

OPENAI_VERSION = tuple([int(x) for x in openai.version.VERSION.split(".")])
TEST_EMBEDDINGS_MODEL = "text-embedding-3-small"
TEST_EMBEDDINGS_INPUT = "South Atlantic Ocean."


@pytest.mark.vcr()
def test_embeddings(default_openai_env, trace_exporter, metrics_reader):
    client = openai.OpenAI()

    response = client.embeddings.create(model=TEST_EMBEDDINGS_MODEL, input=[TEST_EMBEDDINGS_INPUT])

    assert len(response.data) == 1

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {TEST_EMBEDDINGS_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_USAGE_INPUT_TOKENS: 4,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }
    assert span.events == ()

    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_EMBEDDINGS_MODEL,
    }
    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    assert_operation_duration_metric(
        client, "embeddings", operation_duration_metric, attributes=attributes, min_data_point=0.2263190783560276
    )
    assert_token_usage_input_metric(client, "embeddings", token_usage_metric, attributes=attributes, input_data_point=4)


@pytest.mark.vcr()
def test_embeddings_all_the_client_options(default_openai_env, trace_exporter, metrics_reader):
    client = openai.OpenAI()

    response = client.embeddings.create(
        model=TEST_EMBEDDINGS_MODEL, input=[TEST_EMBEDDINGS_INPUT], encoding_format="float"
    )

    assert len(response.data) == 1

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {TEST_EMBEDDINGS_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_REQUEST_ENCODING_FORMATS: ("float",),
        GEN_AI_USAGE_INPUT_TOKENS: 4,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_EMBEDDINGS_MODEL,
    }
    assert_operation_duration_metric(
        client, "embeddings", operation_duration_metric, attributes=attributes, min_data_point=0.2263190783560276
    )
    assert_token_usage_input_metric(
        client,
        "embeddings",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
    )


@pytest.mark.skipif(OPENAI_VERSION < (1, 13, 4), reason="openai.NOT_GIVEN not available")
@pytest.mark.vcr()
def test_embeddings_all_the_client_options_not_given(default_openai_env, trace_exporter, metrics_reader):
    client = openai.OpenAI()

    response = client.embeddings.create(
        model=TEST_EMBEDDINGS_MODEL, input=[TEST_EMBEDDINGS_INPUT], encoding_format=openai.NOT_GIVEN
    )

    assert len(response.data) == 1

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {TEST_EMBEDDINGS_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_USAGE_INPUT_TOKENS: 4,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_EMBEDDINGS_MODEL,
    }
    assert_operation_duration_metric(
        client, "embeddings", operation_duration_metric, attributes=attributes, min_data_point=0.050556943751871586
    )
    assert_token_usage_input_metric(
        client,
        "embeddings",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
    )


@pytest.mark.integration
def test_embeddings_all_the_client_options_integration(trace_exporter, metrics_reader):
    client = get_integration_client()
    model = os.getenv("TEST_EMBEDDINGS_MODEL", TEST_EMBEDDINGS_MODEL)

    response = client.embeddings.create(model=model, input=[TEST_EMBEDDINGS_INPUT], encoding_format="float")

    assert len(response.data) == 1

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_MODEL: response.model,
        GEN_AI_REQUEST_ENCODING_FORMATS: ("float",),
        GEN_AI_USAGE_INPUT_TOKENS: response.usage.prompt_tokens,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response.model,
    }
    assert_operation_duration_metric(
        client, "embeddings", operation_duration_metric, attributes=attributes, min_data_point=MOCK_POSITIVE_FLOAT
    )
    assert_token_usage_input_metric(
        client, "embeddings", token_usage_metric, attributes=attributes, input_data_point=response.usage.prompt_tokens
    )


def test_embeddings_connection_error(trace_exporter, metrics_reader):
    client = openai.Client(base_url="http://localhost:9999/v5", api_key="text-embedding-3-large", max_retries=1)

    with pytest.raises(Exception):
        client.embeddings.create(model=TEST_EMBEDDINGS_MODEL, input=[TEST_EMBEDDINGS_INPUT])

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {TEST_EMBEDDINGS_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_SYSTEM: "openai",
        ERROR_TYPE: "APIConnectionError",
        SERVER_ADDRESS: "localhost",
        SERVER_PORT: 9999,
    }
    assert span.events == ()

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        ERROR_TYPE: "APIConnectionError",
    }
    assert_error_operation_duration_metric(
        "embeddings",
        operation_duration_metric,
        attributes=attributes,
        data_point=0.460242404602468,
        value_delta=1.0,
    )


@pytest.mark.vcr(cassette_name="test_embeddings.yaml")
@pytest.mark.asyncio
async def test_embeddings_async(default_openai_env, trace_exporter, metrics_reader):
    client = openai.AsyncOpenAI()

    response = await client.embeddings.create(model=TEST_EMBEDDINGS_MODEL, input=[TEST_EMBEDDINGS_INPUT])

    assert len(response.data) == 1

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {TEST_EMBEDDINGS_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_USAGE_INPUT_TOKENS: 4,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_EMBEDDINGS_MODEL,
    }
    assert_operation_duration_metric(
        client, "embeddings", operation_duration_metric, attributes=attributes, min_data_point=0.2263190783560276
    )
    assert_token_usage_input_metric(
        client,
        "embeddings",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
    )


@pytest.mark.vcr(cassette_name="test_embeddings_all_the_client_options.yaml")
@pytest.mark.asyncio
async def test_embeddings_async_all_the_client_options(default_openai_env, trace_exporter, metrics_reader):
    client = openai.AsyncOpenAI()

    response = await client.embeddings.create(
        model=TEST_EMBEDDINGS_MODEL, input=[TEST_EMBEDDINGS_INPUT], encoding_format="float"
    )

    assert len(response.data) == 1

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {TEST_EMBEDDINGS_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_REQUEST_ENCODING_FORMATS: ("float",),
        GEN_AI_USAGE_INPUT_TOKENS: 4,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_EMBEDDINGS_MODEL,
    }
    assert_operation_duration_metric(
        client, "embeddings", operation_duration_metric, attributes=attributes, min_data_point=0.2263190783560276
    )
    assert_token_usage_input_metric(
        client,
        "embeddings",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embeddings_async_all_the_client_options_integration(trace_exporter, metrics_reader):
    client = get_integration_async_client()
    model = os.getenv("TEST_EMBEDDINGS_MODEL", TEST_EMBEDDINGS_MODEL)

    params = {
        "model": model,
        "input": [TEST_EMBEDDINGS_INPUT],
        "encoding_format": "float",
    }

    response = await client.embeddings.create(**params)

    assert len(response.data) == 1

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    expected_attrs = {
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_MODEL: response.model,
        GEN_AI_REQUEST_ENCODING_FORMATS: ("float",),
        GEN_AI_USAGE_INPUT_TOKENS: response.usage.prompt_tokens,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }
    assert dict(span.attributes) == expected_attrs

    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response.model,
    }
    assert_operation_duration_metric(
        client, "embeddings", operation_duration_metric, attributes=attributes, min_data_point=MOCK_POSITIVE_FLOAT
    )
    assert_token_usage_input_metric(
        client, "embeddings", token_usage_metric, attributes=attributes, input_data_point=response.usage.prompt_tokens
    )


@pytest.mark.asyncio
async def test_embeddings_async_connection_error(default_openai_env, trace_exporter, metrics_reader):
    client = openai.AsyncOpenAI(base_url="http://localhost:9999/v5", api_key="unused", max_retries=1)

    with pytest.raises(Exception):
        await client.embeddings.create(model=TEST_EMBEDDINGS_MODEL, input=[TEST_EMBEDDINGS_INPUT])

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"embeddings {TEST_EMBEDDINGS_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        GEN_AI_SYSTEM: "openai",
        ERROR_TYPE: "APIConnectionError",
        SERVER_ADDRESS: "localhost",
        SERVER_PORT: 9999,
    }

    assert span.events == ()

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_EMBEDDINGS_MODEL,
        ERROR_TYPE: "APIConnectionError",
    }
    assert_error_operation_duration_metric(
        "embeddings",
        operation_duration_metric,
        attributes=attributes,
        data_point=0.2263190783560276,
        value_delta=1.0,
    )


@pytest.mark.vcr()
def test_embeddings_without_model_parameter(default_openai_env, trace_exporter, metrics_reader):
    client = openai.OpenAI()

    with pytest.raises(TypeError, match=re.escape("create() missing 1 required keyword-only argument: 'model'")):
        client.embeddings.create(input=[TEST_EMBEDDINGS_INPUT])

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "embeddings"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        ERROR_TYPE: "TypeError",
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_SYSTEM: "openai",
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    attributes = {
        "error.type": "TypeError",
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
        "gen_ai.operation.name": "embeddings",
    }
    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    assert_error_operation_duration_metric(
        "embeddings", operation_duration_metric, attributes=attributes, data_point=4.2263190783560276, value_delta=5
    )


@pytest.mark.vcr()
def test_embeddings_model_not_found(default_openai_env, trace_exporter, metrics_reader):
    # force a timeout to don't slow down tests
    client = openai.OpenAI(timeout=1)

    exception = openai.NotFoundError
    with pytest.raises(
        exception, match=re.escape("The model `not-found-model` does not exist or you do not have access to it.")
    ):
        client.embeddings.create(model="not-found-model", input=[TEST_EMBEDDINGS_INPUT])

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "embeddings not-found-model"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        ERROR_TYPE: exception.__qualname__,
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_REQUEST_MODEL: "not-found-model",
        GEN_AI_SYSTEM: "openai",
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    attributes = {
        "error.type": exception.__qualname__,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
        "gen_ai.operation.name": "embeddings",
        "gen_ai.request.model": "not-found-model",
    }
    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    assert_error_operation_duration_metric(
        "embeddings", operation_duration_metric, attributes=attributes, data_point=0.05915193818509579
    )
