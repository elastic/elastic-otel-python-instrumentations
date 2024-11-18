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
from .utils import MOCK_POSITIVE_FLOAT, get_sorted_metrics, logrecords_from_logs

OPENAI_VERSION = tuple([int(x) for x in openai.version.VERSION.split(".")])

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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
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


test_multiple_choices_capture_content_log_events_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-ANeSeH4fwAhE7sU211OIx0aI6K16o",
        24,
        8,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-ANeSfhGk22jizSPywSdR4MGCOdc76",
        24,
        8,
        0.002889830619096756,
    ),
    # ollama does not support n>1
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_multiple_choices_capture_content_log_events_test_data,
)
def test_multiple_choices_with_capture_content_log_events(
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
    logs_exporter,
    request,
):
    provider = request.getfixturevalue(provider_str)

    client = provider.get_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true", "ELASTIC_OTEL_GENAI_EVENTS": "log"}
    ):
        OpenAIInstrumentor().instrument()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    chat_completion = client.chat.completions.create(model=model, messages=messages, n=2)

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
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop", "stop"),
        GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 3
    log_records = logrecords_from_logs(logs)
    user_message, choice, second_choice = log_records
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {"content": "Answer in up to 3 words: Which ocean contains the falkland islands?"}
    assert choice.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    expected_body = {
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": content,
        },
    }
    assert dict(choice.body) == expected_body

    assert second_choice.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    second_expected_body = {
        "finish_reason": "stop",
        "index": 1,
        "message": {
            "content": content,
        },
    }
    assert dict(second_choice.body) == second_expected_body

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
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
        "call_yU31CceO4OliQuZgPdsSPRXT",
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
        "call_EOPkr9g0pp71Wt6LEteCHWkZ",
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
        "call_sr2j6oa1",
        241,
        28,
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,function_call_id,input_tokens,output_tokens,duration",
    test_tools_with_capture_content_test_data,
)
def test_tools_with_capture_content(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    function_call_id,
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
        "gen_ai.completion": '[{"role": "assistant", "content": "", "tool_calls": [{"id": "'
        + function_call_id
        + '", "type": "function", "function": {"name": "get_delivery_date", "arguments": "{\\"order_id\\":\\"order_12345\\"}"}}]}]'
    }

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


test_tools_with_capture_content_log_events_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AIEV9MTVUJ4HtPJm6pro1FgWlWQ2g",
        "call_jQEwfwLUeVLVsxWaz8N0c8Yp",
        140,
        19,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean",
        "chatcmpl-AIEVA7wwx1Cxy8O31zWDnxgUETAeN",
        "call_Xyr5OWcqhvuW62vUx3sPPruH",
        140,
        19,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falklands Islands are located in the oceans south of South America.",
        "chatcmpl-339",
        "call_3h40tlh2",
        241,
        28,
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,function_call_id,input_tokens,output_tokens,duration",
    test_tools_with_capture_content_log_events_test_data,
)
def test_tools_with_capture_content_log_events(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    function_call_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    logs_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true", "ELASTIC_OTEL_GENAI_EVENTS": "log"}
    ):
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 5
    log_records = logrecords_from_logs(logs)
    system_message, user_message, assistant_message, second_user_message, choice = log_records
    assert system_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.system.message"}
    assert system_message.body == {
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."
    }
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {"content": "Hi, can you tell me the delivery date for my order?"}
    assert assistant_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.assistant.message"}
    assert assistant_message.body == {
        "content": "Hi there! I can help with that. Can you please provide your order ID?"
    }
    assert second_user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert second_user_message.body == {"content": "i think it is order_12345"}
    assert choice.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    expected_body = {
        "finish_reason": "tool_calls",
        "index": 0,
        "message": {
            "tool_calls": [
                {
                    "function": {"arguments": '{"order_id":"order_12345"}', "name": "get_delivery_date"},
                    "id": function_call_id,
                    "type": "function",
                },
            ],
        },
    }
    assert dict(choice.body) == expected_body

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "provider_str,model,response_model",
    [
        (
            "openai_provider_chat_completions",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
        ),
    ],
)
def test_tools_with_capture_content_log_events_integration(
    provider_str,
    model,
    response_model,
    trace_exporter,
    logs_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true", "ELASTIC_OTEL_GENAI_EVENTS": "log"}
    ):
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
        GEN_AI_RESPONSE_ID: response.id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
        GEN_AI_USAGE_INPUT_TOKENS: response.usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: response.usage.completion_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 5
    log_records = logrecords_from_logs(logs)
    system_message, user_message, assistant_message, second_user_message, choice = log_records
    assert system_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.system.message"}
    assert system_message.body == {
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."
    }
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {"content": "Hi, can you tell me the delivery date for my order?"}
    assert assistant_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.assistant.message"}
    assert assistant_message.body == {
        "content": "Hi there! I can help with that. Can you please provide your order ID?"
    }
    assert second_user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert second_user_message.body == {"content": "i think it is order_12345"}
    assert choice.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    expected_body = {
        "finish_reason": "tool_calls",
        "index": 0,
        "message": {
            "tool_calls": [
                {
                    "function": {"arguments": '{"order_id":"order_12345"}', "name": "get_delivery_date"},
                    "id": tool_call.id,
                    "type": "function",
                },
            ],
        },
    }
    assert dict(choice.body) == expected_body

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=MOCK_POSITIVE_FLOAT
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=response.usage.prompt_tokens,
        output_data_point=response.usage.completion_tokens,
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
        0.971050308085978,
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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "provider_str,model,response_model",
    [
        (
            "openai_provider_chat_completions",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
        )
    ],
)
def test_basic_with_capture_content_integration(
    provider_str,
    model,
    response_model,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    client = provider.get_client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    assert content

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
        GEN_AI_RESPONSE_ID: response.id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: response.usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: response.usage.completion_tokens,
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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=MOCK_POSITIVE_FLOAT
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=response.usage.prompt_tokens,
        output_data_point=response.usage.completion_tokens,
    )


test_basic_with_capture_content_log_events_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "Atlantic Ocean.",
        "chatcmpl-AIEVEddriZ8trWDORY6MdqNgqRkDX",
        24,
        3,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-AIEVEmwAmuw8qGX1PViCfm0kTe9O8",
        24,
        4,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "Atlantic Ocean",
        "chatcmpl-694",
        46,
        3,
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_basic_with_capture_content_log_events_test_data,
)
def test_basic_with_capture_content_log_events(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    logs_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true", "ELASTIC_OTEL_GENAI_EVENTS": "log"}
    ):
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {"content": "Answer in up to 3 words: Which ocean contains the falkland islands?"}
    assert dict(choice.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    expected_body = {
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": content,
        },
    }
    assert dict(choice.body) == expected_body

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )


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


@pytest.mark.skipif(OPENAI_VERSION < (1, 26, 0), reason="stream_options added in 1.26.0")
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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


@pytest.mark.skipif(OPENAI_VERSION < (1, 26, 0), reason="stream_options added in 1.26.0")
@pytest.mark.integration
@pytest.mark.parametrize(
    "provider_str,model,response_model",
    [
        (
            "openai_provider_chat_completions",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
        )
    ],
)
def test_stream_with_include_usage_option_and_capture_content_integration(
    provider_str,
    model,
    response_model,
    trace_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    client = provider.get_client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    response = client.chat.completions.create(
        model=model, messages=messages, stream=True, stream_options={"include_usage": True}
    )
    chunks = [chunk for chunk in response]
    usage = chunks[-1].usage

    chunks_content = [chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices]
    content = "".join(chunks_content)
    assert content

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
        GEN_AI_RESPONSE_ID: chunks[0].id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: usage.completion_tokens,
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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=MOCK_POSITIVE_FLOAT
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=usage.prompt_tokens,
        output_data_point=usage.completion_tokens,
    )


test_stream_with_tools_and_capture_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "",
        {
            "gen_ai.completion": '[{"role": "assistant", "content": "", "tool_calls": [{"function": {"arguments": "{\\"order_id\\":\\"order_12345\\"}", "name": "get_delivery_date"}, "id": "call_Hb0tFTds0zHfckULpnZ4t8XL", "type": "function"}]}]'
        },
        "chatcmpl-AEGTFZ2zBPeLJlZ1EA10ZEDA12VfO",
        "tool_calls",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "",
        {
            "gen_ai.completion": '[{"role": "assistant", "content": "", "tool_calls": [{"function": {"arguments": "{\\"order_id\\":\\"order_12345\\"}", "name": "get_delivery_date"}, "id": "call_96LHcDPBXtgIxgxbFvuJjTYU", "type": "function"}]}]'
        },
        "chatcmpl-AEGTHbbvqBK2BE52I8GByDCM2dypS",
        "tool_calls",
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        '<tool_call>\n{"name": "get_delivery_date", "arguments": {"order_id": "order_12345"}}\n</tool_call>',
        {
            "gen_ai.completion": '[{"role": "assistant", "content": '
            + json.dumps(
                '<tool_call>\n{"name": "get_delivery_date", "arguments": {"order_id": "order_12345"}}\n</tool_call>'
            )
            + "}]"
        },
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
    assert dict(completion_event.attributes) == completion_content

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )


test_stream_with_tools_and_capture_content_log_events_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "",
        '{"order_id": "order_12345"}',
        "chatcmpl-AIEVFr8IGqjRC2wxrGU3tcRjNbGKf",
        "tool_calls",
        "call_BQ6tpzuq28epoO6jzUNSdG6r",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "",
        '{"order_id": "order_12345"}',
        "chatcmpl-AIEVHRdUM6ip3Uolr8CcrlGhchugq",
        "tool_calls",
        "call_XNHRbrreMnt9ReHJfNH30mom",
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
        "chatcmpl-436",
        "stop",
        "ciao",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,completion_content,response_id,finish_reason,function_call_id,duration",
    test_stream_with_tools_and_capture_content_log_events_test_data,
)
def test_stream_with_tools_and_capture_content_log_events(
    provider_str,
    model,
    response_model,
    content,
    completion_content,
    response_id,
    finish_reason,
    function_call_id,
    duration,
    trace_exporter,
    logs_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true", "ELASTIC_OTEL_GENAI_EVENTS": "log"}
    ):
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 5
    log_records = logrecords_from_logs(logs)
    system_message, user_message, assistant_message, second_user_message, choice = log_records
    assert system_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.system.message"}
    assert system_message.body == {
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."
    }
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {"content": "Hi, can you tell me the delivery date for my order?"}
    assert assistant_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.assistant.message"}
    assert assistant_message.body == {
        "content": "Hi there! I can help with that. Can you please provide your order ID?"
    }
    assert second_user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert second_user_message.body == {"content": "i think it is order_12345"}
    assert choice.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    if finish_reason == "tool_calls":
        expected_body = {
            "finish_reason": finish_reason,
            "index": 0,
            "message": {
                "tool_calls": [
                    {
                        "function": {"arguments": '{"order_id":"order_12345"}', "name": "get_delivery_date"},
                        "id": function_call_id,
                        "type": "function",
                    },
                ]
            },
        }
    else:
        expected_body = {
            "finish_reason": finish_reason,
            "index": 0,
            "message": {
                "content": content,
            },
        }
    assert dict(choice.body) == expected_body

    span_ctx = span.get_span_context()
    assert choice.trace_id == span_ctx.trace_id
    assert choice.span_id == span_ctx.span_id
    assert choice.trace_flags == span_ctx.trace_flags

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )


# Azure is not tested because only gpt-4o version 2024-08-06 supports structured output:
# openai.BadRequestError: Error code: 400 - {'error': {'message': 'Structured output is not allowed.', 'type': 'invalid_request_error', 'param': None, 'code': None}}
test_stream_with_parallel_tools_and_capture_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "",
        {
            "gen_ai.completion": '[{"role": "assistant", "content": "", "tool_calls": [{"function": {"arguments": "{\\"location\\": \\"New York\\"}", "name": "get_weather"}, "id": "call_uDsEOSTauJkgNI8ciF0AvU0X", "type": "function"}, {"function": {"arguments": "{\\"location\\": \\"London\\"}", "name": "get_weather"}, "id": "call_If8zqvcIX9JYzEYJ02dlpoBX", "type": "function"}]}]'
        },
        "chatcmpl-AGooCqGLiGVX21z77Wlic8pSD93XP",
        "tool_calls",
        0.006761051714420319,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        'To provide you with the most current information about the weather, I need to know which cities you are interested in. Could we please specify the name of each city? For instance, both "New York", "London" or individual names like "New York City".',
        {
            "gen_ai.completion": '[{"role": "assistant", "content": "To provide you with the most current information about the weather, I need to know which cities you are interested in. Could we please specify the name of each city? For instance, both \\"New York\\", \\"London\\" or individual names like \\"New York City\\"."}]'
        },
        "chatcmpl-142",
        "stop",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,completion_content,response_id,finish_reason,duration",
    test_stream_with_parallel_tools_and_capture_content_test_data,
)
def test_stream_with_parallel_tools_and_capture_content(
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
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant providing weather updates.",
        },
        {"role": "user", "content": "What is the weather in New York City and London?"},
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
    assert dict(completion_event.attributes) == completion_content

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )


# Azure is not tested because only gpt-4o version 2024-08-06 supports structured output:
# openai.BadRequestError: Error code: 400 - {'error': {'message': 'Structured output is not allowed.', 'type': 'invalid_request_error', 'param': None, 'code': None}}
test_stream_with_parallel_tools_and_capture_content_log_events_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "",
        json.dumps(""),
        "chatcmpl-AICov4avW9uwU1rfxlUzPKGG5BiCs",
        "tool_calls",
        0.006761051714420319,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "<tool_call>\n"
        + json.dumps({"name": "get_weather", "arguments": {"location": "New York, NY"}}, indent=2)
        + "\n</tool_call>\n<tool_call>\n"
        + json.dumps({"name": "get_weather", "arguments": {"location": "London, UK"}}, indent=2)
        + "\n</tool_call>",
        '{"message": {"content":"<tool_call>\n'
        + json.dumps({"name": "get_weather", "arguments": {"location": "New York, NY"}}, indent=2)
        + "\n</tool_call>\n<tool_call>\n"
        + json.dumps({"name": "get_weather", "arguments": {"location": "London, UK"}}, indent=2)
        + "\n</tool_call>",
        "chatcmpl-986",
        "stop",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,completion_content,response_id,finish_reason,duration",
    test_stream_with_parallel_tools_and_capture_content_log_events_test_data,
)
def test_stream_with_parallel_tools_and_capture_content_log_events(
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
    logs_exporter,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true", "ELASTIC_OTEL_GENAI_EVENTS": "log"}
    ):
        OpenAIInstrumentor().instrument()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant providing weather updates.",
        },
        {"role": "user", "content": "What is the weather in New York City and London?"},
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 3
    log_records = logrecords_from_logs(logs)
    system_message, user_message, choice = log_records
    assert system_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.system.message"}
    assert system_message.body == {"content": "You are a helpful assistant providing weather updates."}
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {"content": "What is the weather in New York City and London?"}
    assert choice.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    if finish_reason == "tool_calls":
        expected_body = {
            "finish_reason": finish_reason,
            "index": 0,
            "message": {
                "tool_calls": [
                    {
                        "function": {"arguments": '{"location": "New York City"}', "name": "get_weather"},
                        "id": "call_m8FzMvtVd3wjksWMeRCWkPDK",
                        "type": "function",
                    },
                    {
                        "function": {"arguments": '{"location": "London"}', "name": "get_weather"},
                        "id": "call_4WcXUPtB1wlKUy1lOrguqAtC",
                        "type": "function",
                    },
                ]
            },
        }
    else:
        expected_body = {
            "finish_reason": finish_reason,
            "index": 0,
            "message": {
                "content": content,
            },
        }
    assert dict(choice.body) == expected_body

    span_ctx = span.get_span_context()
    assert choice.trace_id == span_ctx.trace_id
    assert choice.span_id == span_ctx.span_id
    assert choice.trace_flags == span_ctx.trace_flags

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )


test_tools_with_followup_and_capture_content_log_events_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        None,
        json.dumps(""),
        "tool_calls",
        0.007433261722326279,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        None,
        json.dumps(""),
        "tool_calls",
        0.003254897892475128,
    ),
    # ollama does not return tool calls
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,completion_content,finish_reason,duration",
    test_tools_with_followup_and_capture_content_log_events_test_data,
)
def test_tools_with_followup_and_capture_content_log_events(
    provider_str,
    model,
    response_model,
    content,
    completion_content,
    finish_reason,
    duration,
    trace_exporter,
    metrics_reader,
    logs_exporter,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true", "ELASTIC_OTEL_GENAI_EVENTS": "log"}
    ):
        OpenAIInstrumentor().instrument()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant providing weather updates.",
        },
        {"role": "user", "content": "What is the weather in New York City and London?"},
    ]

    first_response = client.chat.completions.create(model=model, messages=messages, tools=tools)

    assert first_response.choices[0].message.content == content

    first_reponse_message = first_response.choices[0].message
    if hasattr(first_reponse_message, "to_dict"):
        previous_message = first_response.choices[0].message.to_dict()
    else:
        # old pydantic from old openai client
        previous_message = first_response.choices[0].message.model_dump()
    followup_messages = [
        {
            "role": "assistant",
            "tool_calls": previous_message["tool_calls"],
        },
        {
            "role": "tool",
            "content": "25 degrees and sunny",
            "tool_call_id": previous_message["tool_calls"][0]["id"],
        },
        {
            "role": "tool",
            "content": "15 degrees and raining",
            "tool_call_id": previous_message["tool_calls"][1]["id"],
        },
    ]

    second_response = client.chat.completions.create(model=model, messages=messages + followup_messages)

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 2

    first_span, second_span = spans
    assert first_span.name == f"chat {model}"
    assert first_span.kind == SpanKind.CLIENT
    assert first_span.status.status_code == StatusCode.UNSET

    assert dict(first_span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: first_response.id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: (finish_reason,),
        GEN_AI_USAGE_INPUT_TOKENS: first_response.usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: first_response.usage.completion_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }

    assert second_span.name == f"chat {model}"
    assert second_span.kind == SpanKind.CLIENT
    assert second_span.status.status_code == StatusCode.UNSET

    assert dict(second_span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: second_response.id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: second_response.usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: second_response.usage.completion_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 9
    log_records = logrecords_from_logs(logs)

    # first call events
    system_message, user_message, choice = log_records[:3]
    assert system_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.system.message"}
    assert system_message.body == {"content": "You are a helpful assistant providing weather updates."}
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {"content": "What is the weather in New York City and London?"}
    assert choice.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    expected_body = {
        "finish_reason": finish_reason,
        "index": 0,
        "message": {
            "tool_calls": [
                {
                    "function": {"arguments": '{"location": "New York City"}', "name": "get_weather"},
                    "id": previous_message["tool_calls"][0]["id"],
                    "type": "function",
                },
                {
                    "function": {"arguments": '{"location": "London"}', "name": "get_weather"},
                    "id": previous_message["tool_calls"][1]["id"],
                    "type": "function",
                },
            ]
        },
    }
    assert dict(choice.body) == expected_body

    # second call events
    system_message, user_message, assistant_message, first_tool, second_tool, choice = log_records[3:]
    assert system_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.system.message"}
    assert system_message.body == {"content": "You are a helpful assistant providing weather updates."}
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {"content": "What is the weather in New York City and London?"}
    assert assistant_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.assistant.message"}
    assert assistant_message.body == {"tool_calls": previous_message["tool_calls"]}
    assert first_tool.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.tool.message"}
    first_tool_response = previous_message["tool_calls"][0]
    assert first_tool.body == {"content": "25 degrees and sunny", "id": first_tool_response["id"]}
    assert second_tool.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.tool.message"}
    second_tool_response = previous_message["tool_calls"][1]
    assert second_tool.body == {"content": "15 degrees and raining", "id": second_tool_response["id"]}
    assert choice.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}
    assert choice.body == {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": second_response.choices[0].message.content},
    }

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration, count=2
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=[first_response.usage.prompt_tokens, second_response.usage.prompt_tokens],
        output_data_point=[first_response.usage.completion_tokens, second_response.usage.completion_tokens],
        count=2,
    )


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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


test_async_basic_with_capture_content_log_events_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AIEVJxmbMtOSyVurjk73oqE0uQhAX",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-AIEVKUfan2sD0ScQLHPSMYn7Fet3Y",
        24,
        4,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falkland Islands are located within the southern waters of the South Atlantic Ocean.",
        "chatcmpl-472",
        46,
        17,
        0.002600736916065216,
    ),
]


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_async_basic_with_capture_content_log_events_test_data,
)
async def test_async_basic_with_capture_content_log_events(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    logs_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_async_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true", "ELASTIC_OTEL_GENAI_EVENTS": "log"}
    ):
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {"content": "Answer in up to 3 words: Which ocean contains the falkland islands?"}
    assert dict(choice.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    expected_body = {
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": content,
        },
    }
    assert dict(choice.body) == expected_body

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_str,model,response_model",
    [
        (
            "openai_provider_chat_completions",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
        ),
    ],
)
async def test_async_basic_with_capture_content_log_events_integration(
    provider_str,
    model,
    response_model,
    trace_exporter,
    logs_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_async_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true", "ELASTIC_OTEL_GENAI_EVENTS": "log"}
    ):
        OpenAIInstrumentor().instrument()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    response = await client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    assert content

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
        GEN_AI_RESPONSE_ID: response.id,
        GEN_AI_RESPONSE_MODEL: response_model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: response.usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: response.usage.completion_tokens,
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {"content": "Answer in up to 3 words: Which ocean contains the falkland islands?"}
    assert choice.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    expected_body = {
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": content,
        },
    }
    assert dict(choice.body) == expected_body

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=MOCK_POSITIVE_FLOAT
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=response.usage.prompt_tokens,
        output_data_point=response.usage.completion_tokens,
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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )


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
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )


test_async_stream_with_capture_content_log_events_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-AIEVLVduixLA39qqzjDIfJ2u2dPcJ",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-AIEVNaNWKhuoMnRJtcPUJNjbAz9Ib",
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falkland Islands contain the South Atlantic Ocean.",
        "chatcmpl-466",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,duration",
    test_async_stream_with_capture_content_log_events_test_data,
)
async def test_async_stream_with_capture_content_log_events(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    duration,
    trace_exporter,
    logs_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_async_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true", "ELASTIC_OTEL_GENAI_EVENTS": "log"}
    ):
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {"content": "Answer in up to 3 words: Which ocean contains the falkland islands?"}
    assert dict(choice.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    expected_body = {
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": content,
        },
    }
    assert dict(choice.body) == expected_body

    span_ctx = span.get_span_context()
    assert choice.trace_id == span_ctx.trace_id
    assert choice.span_id == span_ctx.span_id
    assert choice.trace_flags == span_ctx.trace_flags

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )


# FIXME: ollama has empty tool_calls
test_async_tools_with_capture_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "chatcmpl-AEGr0SsAhppNLpPXpTtnmBiGGViQb",
        "call_vZtzXVh5oO3k1IpFfuRejWHv",
        140,
        19,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "chatcmpl-AEGr1X6ieLFOD5hlXZBx2BL2BrSLe",
        "call_46cgdzJzy50oQJwWeUVWIwC3",
        140,
        19,
        0.002889830619096756,
    ),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_str,model,response_model,response_id,function_call_id,input_tokens,output_tokens,duration",
    test_async_tools_with_capture_content_test_data,
)
async def test_async_tools_with_capture_content(
    provider_str,
    model,
    response_model,
    response_id,
    function_call_id,
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
        "gen_ai.completion": '[{"role": "assistant", "content": "", "tool_calls": [{"id": "'
        + function_call_id
        + '", "type": "function", "function": {"name": "get_delivery_date", "arguments": "{\\"order_id\\":\\"order_12345\\"}"}}]}]'
    }

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
    assert_token_usage_metric(
        provider,
        token_usage_metric,
        attributes=attributes,
        input_data_point=input_tokens,
        output_data_point=output_tokens,
    )


# FIXME: ollama has empty tool_calls
test_async_tools_with_capture_content_log_events_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "chatcmpl-AIEVOegIWRWjNVbjxs4iASh4SNKAj",
        "call_n4WJq7bu6UsgjdqeNYxRmGno",
        "",
        140,
        19,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini",
        "chatcmpl-AIEVPAAgrPcEpkQHXamdD9JpNNPep",
        "call_uO2RgeshT5Xmbwg778qN14d5",
        "",
        140,
        19,
        0.002889830619096756,
    ),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_str,model,response_model,response_id,function_call_id,choice_content,input_tokens,output_tokens,duration",
    test_async_tools_with_capture_content_log_events_test_data,
)
async def test_async_tools_with_capture_content_log_events(
    provider_str,
    model,
    response_model,
    response_id,
    function_call_id,
    choice_content,
    input_tokens,
    output_tokens,
    duration,
    trace_exporter,
    logs_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)
    client = provider.get_async_client()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true", "ELASTIC_OTEL_GENAI_EVENTS": "log"}
    ):
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 5
    log_records = logrecords_from_logs(logs)
    system_message, user_message, assistant_message, second_user_message, choice = log_records
    assert dict(system_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.system.message"}
    assert dict(system_message.body) == {
        "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user."
    }
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {"content": "Hi, can you tell me the delivery date for my order?"}
    assert dict(assistant_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.assistant.message"}
    assert dict(assistant_message.body) == {
        "content": "Hi there! I can help with that. Can you please provide your order ID?"
    }
    assert dict(second_user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(second_user_message.body) == {"content": "i think it is order_12345"}
    assert dict(choice.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}

    expected_body = {
        "finish_reason": "tool_calls",
        "index": 0,
        "message": {
            "tool_calls": [
                {
                    "function": {"arguments": '{"order_id":"order_12345"}', "name": "get_delivery_date"},
                    "id": function_call_id,
                    "type": "function",
                },
            ],
        },
    }
    assert dict(choice.body) == expected_body

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )
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


@pytest.mark.vcr()
def test_exported_schema_version(
    ollama_provider_chat_completions,
    trace_exporter,
    metrics_reader,
):
    client = ollama_provider_chat_completions.get_client()

    messages = [
        {
            "role": "user",
            "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
        }
    ]

    client.chat.completions.create(model="qwen2.5:0.5b", messages=messages)

    spans = trace_exporter.get_finished_spans()
    (span,) = spans
    assert span.instrumentation_scope.schema_url == "https://opentelemetry.io/schemas/1.27.0"

    metrics_data = metrics_reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    for metrics in resource_metrics:
        for scope_metrics in metrics.scope_metrics:
            assert scope_metrics.schema_url == "https://opentelemetry.io/schemas/1.27.0"
