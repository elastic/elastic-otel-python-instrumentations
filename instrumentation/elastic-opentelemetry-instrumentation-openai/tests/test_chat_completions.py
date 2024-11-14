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

import json
import re
from unittest import mock

import openai
import pytest
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_SYSTEM,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS, SERVER_PORT

from .conftest import (
    assert_error_operation_duration_metric,
    assert_operation_duration_metric,
    assert_token_usage_metric,
)
from .utils import get_sorted_metrics

providers = ["openai_provider_chat_completions", "ollama_provider_chat_completions", "azure_provider_chat_completions"]


# TODO: provide a wrapper to generate parameters names and values for parametrize?
test_basic_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEEtEV5mndU1WYBSLgvTpAIKAoYeu",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean",
        "chatcmpl-AEEtFi8N27MCQoHr62DvUowImjwEc",
        24,
        3,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falklands Islands are located in the oceans south of South America.",
        "chatcmpl-645",
        46,
        15,
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration", test_basic_test_data
)
def test_basic(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)

    client = provider.get_client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    chat_completion = client.chat.completions.create(model=model, messages=messages)

    assert chat_completion.choices[0].message.content == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


test_all_the_client_options_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEGEFRHEOGpX0H3Nx84zgOEbJ8g6Y",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-AEGHLLqYkgJxUgLgX8RCLeL85irQR",
        24,
        4,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Great British Oceanic Peninsula.",
        "chatcmpl-398",
        46,
        8,
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_all_the_client_options_test_data,
)
def test_all_the_client_options(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        frequency_penalty=0,
        max_tokens=100,  # AzureOpenAI still does not support max_completions_tokens
        presence_penalty=0,
        temperature=1,
        top_p=1,
        stop="foo",
    )

    assert chat_completion.choices[0].message.content == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_FREQUENCY_PENALTY: 0,
        GEN_AI_REQUEST_MAX_TOKENS: 100,
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_REQUEST_PRESENCE_PENALTY: 0,
        GEN_AI_REQUEST_STOP_SEQUENCES: ("foo",),
        GEN_AI_REQUEST_TEMPERATURE: 1,
        GEN_AI_REQUEST_TOP_P: 1,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


test_function_calling_with_tools_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEGIgPL1ReEL2yG6M4MrD3Uw960Bu",
        140,
        19,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean",
        "chatcmpl-AEGIh5ygWzdZJL7BvIVeFLmgDSdT7",
        140,
        19,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falklands Islands are located in the oceans south of South America.",
        "chatcmpl-363",
        241,
        28,
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_function_calling_with_tools_test_data,
)
def test_function_calling_with_tools(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_delivery_date",
                "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The customer's order ID.",
                        },
                    },
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
        },
        {"role": "user", "content": "Hi, can you tell me the delivery date for my order?"},
        {
            "role": "assistant",
            "content": "Hi there! I can help with that. Can you please provide your order ID?",
        },
        {"role": "user", "content": "i think it is order_12345"},
    ]

    response = client.chat.completions.create(model=model, messages=messages, tools=tools)
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_delivery_date"
    # FIXME: add to test data
    assert json.loads(tool_call.function.arguments) == {"order_id": "order_12345"}

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


test_tools_with_capture_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEGM5OhimYEMDsRq20IQCBx4vzf2Z",
        140,
        19,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean",
        "chatcmpl-AEGM6niNxWuulOpOMXNelXUZqF443",
        140,
        19,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falklands Islands are located in the oceans south of South America.",
        "chatcmpl-556",
        241,
        28,
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_tools_with_capture_content_test_data,
)
def test_tools_with_capture_content(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_delivery_date",
                "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The customer's order ID.",
                        },
                    },
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
        },
        {"role": "user", "content": "Hi, can you tell me the delivery date for my order?"},
        {
            "role": "assistant",
            "content": "Hi there! I can help with that. Can you please provide your order ID?",
        },
        {"role": "user", "content": "i think it is order_12345"},
    ]

    response = client.chat.completions.create(model=model, messages=messages, tools=tools)
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_delivery_date"
    assert json.loads(tool_call.function.arguments) == {"order_id": "order_12345"}

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }

    assert len(span.events) == 2
    prompt_event, completion_event = span.events
    assert prompt_event.name == "gen_ai.content.prompt"
    assert dict(prompt_event.attributes) == {"gen_ai.prompt": json.dumps(messages)}
    assert completion_event.name == "gen_ai.content.completion"
    assert dict(completion_event.attributes) == {
        "gen_ai.completion": '[{"role": "assistant", "content": {"order_id": "order_12345"}}]'
    }

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


test_connection_error_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize("provider_str,model,duration", test_connection_error_test_data)
def test_connection_error(provider_str, model, duration, trace_exporter, metrics_reader, request):
    provider = request.getfixturevalue(provider_str)

    client = openai.Client(base_url="http://localhost:9999/v5", api_key="unused", max_retries=1)
    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    with pytest.raises(Exception):
        client.chat.completions.create(model=model, messages=messages)

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
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


test_basic_with_capture_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEFu3fjzje87q8tfrYWpazqelNIfW",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "Atlantic Ocean.",
        "chatcmpl-AEFu4REQubxeCCkrv6wgeJ4VdN6o5",
        24,
        3,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Atlantic Ocean contains the Falkland Islands.",
        "chatcmpl-976",
        46,
        10,
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_basic_with_capture_content_test_data,
)
def test_basic_with_capture_content(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    chat_completion = client.chat.completions.create(model=model, messages=messages)

    assert chat_completion.choices[0].message.content == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }

    assert len(span.events) == 2
    prompt_event, completion_event = span.events
    assert prompt_event.name == "gen_ai.content.prompt"
    assert dict(prompt_event.attributes) == {"gen_ai.prompt": json.dumps(messages)}
    assert completion_event.name == "gen_ai.content.completion"
    assert dict(completion_event.attributes) == {
        "gen_ai.completion": '[{"role": "assistant", "content": "' + content + '"}]'
    }

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


test_stream_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEGTAvX2YSIO9EQwMleHTB91Cgn4G",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-AEGTBkVVthwTc3DgRp6RGKJxyY1pE",
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falkland Islands, also known as the Argentinian Islands or British South America Land (BSAL), is an archipelago of several small islands and peninsulas located off the coast of Argentina. It contains nine uninhabited Falkland Islands, plus a mix of uninhabitable territory and other small features that are not considered part of the Falkland Islands' current administrative divisions.",
        "chatcmpl-415",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize("provider_str,model,response_model,content,response_id,duration", test_stream_test_data)
def test_stream(
    provider_str, model, response_model, content, response_id, duration, trace_exporter, metrics_reader, request
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    chat_completion = client.chat.completions.create(model=model, messages=messages, stream=True)

    chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert span.events == ()

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)


# FIXME: add custom ollama
test_stream_with_include_usage_option_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEGTE6nGqR4tbVuy6CPSHnXIF2eqy",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-AEGTFYMyoBBDm4Qz37Lc8bekb2QOO",
        24,
        4,
        0.002889830619096756,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_stream_with_include_usage_option_test_data,
)
def test_stream_with_include_usage_option(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    chat_completion = client.chat.completions.create(
        model=model, messages=messages, stream=True, stream_options={"include_usage": True}
    )

    chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


test_stream_with_tools_and_capture_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "",
        '{"order_id": "order_12345"}',
        "chatcmpl-AEGTFZ2zBPeLJlZ1EA10ZEDA12VfO",
        "tool_calls",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "",
        '{"order_id": "order_12345"}',
        "chatcmpl-AEGTHbbvqBK2BE52I8GByDCM2dypS",
        "tool_calls",
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        '<tool_call>\n{"name": "get_delivery_date", "arguments": {"order_id": "order_12345"}}\n</tool_call>',
        json.dumps(
            '<tool_call>\n{"name": "get_delivery_date", "arguments": {"order_id": "order_12345"}}\n</tool_call>'
        ),
        "chatcmpl-598",
        "stop",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,completion_content,response_id,finish_reason,duration",
    test_stream_with_tools_and_capture_content_test_data,
)
def test_stream_with_tools_and_capture_content(
    provider_str,
    model,
    response_model,
    content,
    completion_content,
    response_id,
    finish_reason,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_delivery_date",
                "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The customer's order ID.",
                        },
                    },
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
        },
        {"role": "user", "content": "Hi, can you tell me the delivery date for my order?"},
        {
            "role": "assistant",
            "content": "Hi there! I can help with that. Can you please provide your order ID?",
        },
        {"role": "user", "content": "i think it is order_12345"},
    ]

    chat_completion = client.chat.completions.create(model=model, messages=messages, tools=tools, stream=True)

    chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: (finish_reason,),
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }

    assert len(span.events) == 2
    prompt_event, completion_event = span.events
    assert prompt_event.name == "gen_ai.content.prompt"
    assert dict(prompt_event.attributes) == {"gen_ai.prompt": json.dumps(messages)}
    assert completion_event.name == "gen_ai.content.completion"
    assert dict(completion_event.attributes) == {
        "gen_ai.completion": '[{"role": "assistant", "content": ' + completion_content + "}]"
    }

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)


test_async_basic_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEGqstM7lJ74sLzKJxMyhZZJixowh",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-AEGqs0IFKsicvFYSaiyWFpZVr6OHe",
        24,
        4,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "Atlantic Ocean",
        "chatcmpl-425",
        46,
        3,
        0.002600736916065216,
    ),
]


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_async_basic_test_data,
)
async def test_async_basic(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_async_client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    chat_completion = await client.chat.completions.create(model=model, messages=messages)

    assert chat_completion.choices[0].message.content == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert span.events == ()

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


test_async_basic_with_capture_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEGquKdflabT1q3yPFXFggu0hn6Cg",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-AEGqv0J1RJgG477zKEayzIFWtBfoN",
        24,
        4,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "Antarctica",
        "chatcmpl-280",
        46,
        4,
        0.002600736916065216,
    ),
]


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_async_basic_with_capture_content_test_data,
)
async def test_async_basic_with_capture_content(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_async_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    chat_completion = await client.chat.completions.create(model=model, messages=messages)

    assert chat_completion.choices[0].message.content == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert len(span.events) == 2
    prompt_event, completion_event = span.events
    assert prompt_event.name == "gen_ai.content.prompt"
    assert dict(prompt_event.attributes) == {"gen_ai.prompt": json.dumps(messages)}
    assert completion_event.name == "gen_ai.content.completion"
    assert dict(completion_event.attributes) == {
        "gen_ai.completion": ('[{"role": "assistant", "content": "' + content + '"}]')
    }

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


test_async_stream_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEGqwO8cO97nJK1gMyVlzL9Mfyv7E",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-AEGqxYYCkiihSc05nBzHlYVwwUmzl",
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falkland Islands are located on which ocean?",
        "chatcmpl-325",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize("provider_str,model,response_model,content,response_id,duration", test_async_stream_test_data)
async def test_async_stream(
    provider_str, model, response_model, content, response_id, duration, trace_exporter, metrics_reader, request
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_async_client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    chat_completion = await client.chat.completions.create(model=model, messages=messages, stream=True)

    chunks = [chunk.choices[0].delta.content or "" async for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert span.events == ()

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)


test_async_stream_with_capture_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEGqyZRrj9GUzDNw5te55gt1r7eus",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-AEGqzfynQK4iCO7EXRy3kGXYyuxF5",
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falkland Islands lie within which ocean?",
        "chatcmpl-644",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,duration",
    test_async_stream_with_capture_content_test_data,
)
async def test_async_stream_with_capture_content(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_async_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()
    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    chat_completion = await client.chat.completions.create(model=model, messages=messages, stream=True)

    chunks = [chunk.choices[0].delta.content or "" async for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    assert len(span.events) == 2
    prompt_event, completion_event = span.events
    assert prompt_event.name == "gen_ai.content.prompt"
    assert dict(prompt_event.attributes) == {"gen_ai.prompt": json.dumps(messages)}
    assert completion_event.name == "gen_ai.content.completion"
    assert dict(completion_event.attributes) == {
        "gen_ai.completion": ('[{"role": "assistant", "content": "' + content + '"}]')
    }

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)


# FIXME: ollama has empty tool_calls
test_async_tools_with_capture_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AEGr0SsAhppNLpPXpTtnmBiGGViQb",
        140,
        19,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean",
        "chatcmpl-AEGr1X6ieLFOD5hlXZBx2BL2BrSLe",
        140,
        19,
        0.002889830619096756,
    ),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_async_tools_with_capture_content_test_data,
)
async def test_async_tools_with_capture_content(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_async_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_delivery_date",
                "description": "Get the delivery date for a customer's order. Call this whenever you need to know the delivery date, for example when a customer asks 'Where is my package'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The customer's order ID.",
                        },
                    },
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
        },
        {"role": "user", "content": "Hi, can you tell me the delivery date for my order?"},
        {
            "role": "assistant",
            "content": "Hi there! I can help with that. Can you please provide your order ID?",
        },
        {"role": "user", "content": "i think it is order_12345"},
    ]

    response = await client.chat.completions.create(model=model, messages=messages, tools=tools)
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_delivery_date"
    assert json.loads(tool_call.function.arguments) == {"order_id": "order_12345"}

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response_id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }

    assert len(span.events) == 2
    prompt_event, completion_event = span.events
    assert prompt_event.name == "gen_ai.content.prompt"
    assert dict(prompt_event.attributes) == {"gen_ai.prompt": json.dumps(messages)}
    assert completion_event.name == "gen_ai.content.completion"
    assert dict(completion_event.attributes) == {
        "gen_ai.completion": '[{"role": "assistant", "content": {"order_id": "order_12345"}}]'
    }

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(provider, operation_duration_metric, attributes=attributes, data_point=duration)
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


test_without_model_parameter_test_data = [
    (
        "openai_provider_chat_completions",
        "api.openai.com",
        443,
        5,
    ),
    (
        "azure_provider_chat_completions",
        "test.openai.azure.com",
        443,
        5,
    ),
    (
        "ollama_provider_chat_completions",
        "localhost",
        11434,
        5,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize("provider_str,server_address,server_port,duration", test_without_model_parameter_test_data)
def test_without_model_parameter(
    provider_str,
    server_address,
    server_port,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)

    client = provider.get_client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Missing required arguments; Expected either ('messages' and 'model') or ('messages', 'model' and 'stream') arguments to be given"
        ),
    ):
        client.chat.completions.create(messages=messages)

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "chat"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    assert dict(span.attributes) == {
        ERROR_TYPE: "TypeError",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_SYSTEM: "openai",
        SERVER_ADDRESS: server_address,
        SERVER_PORT: server_port,
    }

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {"error.type": "TypeError", "server.address": server_address, "server.port": server_port}
    assert_error_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, data_point=duration, value_delta=5
    )


test_with_model_not_found_test_data = [
    (
        "openai_provider_chat_completions",
        "api.openai.com",
        443,
        "The model `not-found-model` does not exist or you do not have access to it.",
        0.00230291485786438,
    ),
    (
        "azure_provider_chat_completions",
        "test.openai.azure.com",
        443,
        "The API deployment for this resource does not exist. If you created the deployment within the last 5 minutes, please wait a moment and try again.",
        0.00230291485786438,
    ),
    (
        "ollama_provider_chat_completions",
        "localhost",
        11434,
        'model "not-found-model" not found, try pulling it first',
        0.00230291485786438,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,server_address,server_port,exception,duration", test_with_model_not_found_test_data
)
def test_with_model_not_found(
    provider_str,
    server_address,
    server_port,
    exception,
    duration,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)

    client = provider.get_client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    with pytest.raises(openai.NotFoundError, match="Error code: 404.*" + re.escape(exception)):
        client.chat.completions.create(model="not-found-model", messages=messages)

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "chat not-found-model"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    assert dict(span.attributes) == {
        ERROR_TYPE: "NotFoundError",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: "not-found-model",
        GEN_AI_SYSTEM: "openai",
        SERVER_ADDRESS: server_address,
        SERVER_PORT: server_port,
    }

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        "gen_ai.request.model": "not-found-model",
        "error.type": "NotFoundError",
        "server.address": server_address,
        "server.port": server_port,
    }
    assert_error_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, data_point=duration
    )
