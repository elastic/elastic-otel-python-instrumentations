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
from dataclasses import dataclass
from typing import List
from unittest import mock

import openai
import pytest
from opentelemetry._events import Event
from opentelemetry._logs import LogRecord
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT,
    GEN_AI_OPENAI_REQUEST_SEED,
    GEN_AI_OPENAI_REQUEST_SERVICE_TIER,
    GEN_AI_OPENAI_RESPONSE_SERVICE_TIER,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS, SERVER_PORT
from opentelemetry.trace import SpanKind, StatusCode

from .conftest import (
    assert_error_operation_duration_metric,
    assert_operation_duration_metric,
    assert_token_usage_metric,
)
from .utils import MOCK_POSITIVE_FLOAT, get_sorted_metrics, logrecords_from_logs

OPENAI_VERSION = tuple([int(x) for x in openai.version.VERSION.split(".")])


# TODO: provide a wrapper to generate parameters names and values for parametrize?
test_basic_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-ASfa6PzPwTdpKRmM5equ6f0yxNbkr",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "Atlantic Ocean",
        "chatcmpl-ASxkAADVoG0VU3xx1Q0D1afSVpqLQ",
        24,
        2,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Atlantic Ocean.",
        "chatcmpl-126",
        46,
        5,
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
    logs_exporter,
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice, expected_content=None)

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
        "chatcmpl-Ab7DgqASxxcxoeuHYGRPqWaL6rJok",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-Ab7DhFk7vSvmMW4ICIZh0gkvTZn7G",
        24,
        4,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "Amalfis Sea",
        "chatcmpl-593",
        46,
        5,
        0.002600736916065216,
    ),
]


@pytest.mark.skipif(OPENAI_VERSION < (1, 35, 0), reason="service tieri added in 1.35.0")
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
    logs_exporter,
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
        seed=100,
        service_tier="default",
        response_format={"type": "text"},
    )

    assert chat_completion.choices[0].message.content == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    expected_attrs = {
        GEN_AI_OPENAI_REQUEST_SEED: 100,
        GEN_AI_OPENAI_REQUEST_SERVICE_TIER: "default",
        GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT: "text",
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
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
    if provider_str != "openai_provider_chat_completions":
        del expected_attrs[GEN_AI_OPENAI_RESPONSE_SERVICE_TIER]
    assert dict(span.attributes) == expected_attrs

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice, expected_content=None)

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


test_multiple_choices_capture_message_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-ASfa8r4rkn4OQSmqDqjdHf2UtP4Gn",
        24,
        8,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "Atlantic Ocean",
        "chatcmpl-ASxkC12M7RgRDgP5GjnYlUwEWQEDo",
        24,
        4,
        0.002889830619096756,
    ),
    # ollama does not support n>1
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_multiple_choices_capture_message_content_test_data,
)
def test_multiple_choices_with_capture_message_content(
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
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
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

    assert_stop_log_record(choice, content)
    assert_stop_log_record(second_choice, content, 1)

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
        "chatcmpl-ASfa8hgDJKumHgFlD27gZDNSC8HzZ",
        "call_62Px1tSvshkL8RBrj4p4msgO",
        140,
        19,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "South Atlantic Ocean",
        "chatcmpl-ASxkDInQTANJ57p0VfUCuAISgNbW8",
        "call_U0QYBadhpy4pBO6jYPm09KvZ",
        144,
        20,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falklands Islands are located in the oceans south of South America.",
        "chatcmpl-641",
        "call_ww759p36",
        241,
        28,
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,function_call_id,input_tokens,output_tokens,duration",
    test_function_calling_with_tools_test_data,
)
def test_function_calling_with_tools(
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
    logs_exporter,
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 5
    log_records = logrecords_from_logs(logs)
    system_message, user_message, assistant_message, second_user_message, choice = log_records
    assert system_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.system.message"}
    assert system_message.body == {}
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {}
    assert assistant_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.assistant.message"}
    assert assistant_message.body == {}
    assert second_user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert second_user_message.body == {}

    assert_tool_call_log_record(
        choice, [ToolCall(function_call_id, "get_delivery_date", '{"order_id": "order_12345"}')]
    )

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


test_tools_with_capture_message_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-ASfaAdUvnmifbYTNRZYh0TbM7mmTu",
        "call_pWCLNanMRK7W7uEVHK8PzIKU",
        140,
        19,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "South Atlantic Ocean",
        "chatcmpl-ASxkGiNWw960EEmcHpv6CgIyP6tHy",
        "call_hR2GEOnGJhmLsHHsMgfLuICf",
        144,
        20,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falklands Islands are located in the oceans south of South America.",
        "chatcmpl-695",
        "call_stzzh47r",
        241,
        28,
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,function_call_id,input_tokens,output_tokens,duration",
    test_tools_with_capture_message_content_test_data,
)
def test_tools_with_capture_message_content(
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
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
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

    assert_tool_call_log_record(
        choice, [ToolCall(function_call_id, "get_delivery_date", '{"order_id": "order_12345"}')]
    )

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
def test_tools_with_capture_message_content_integration(
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
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
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

    assert_tool_call_log_record(choice, [ToolCall(tool_call.id, "get_delivery_date", '{"order_id": "order_12345"}')])

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
        1.026234219999992,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        0.971050308085978,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        1.0064430559999948,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize("provider_str,model,duration", test_connection_error_test_data)
def test_connection_error(provider_str, model, duration, trace_exporter, metrics_reader, logs_exporter, request):
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 1
    log_records = logrecords_from_logs(logs)
    (user_message,) = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

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
def test_basic_with_capture_message_content_integration(
    provider_str,
    model,
    response_model,
    trace_exporter,
    logs_exporter,
    metrics_reader,
    request,
):
    provider = request.getfixturevalue(provider_str)

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"},
    ):
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {"content": "Answer in up to 3 words: Which ocean contains the falkland islands?"}

    assert_stop_log_record(choice, content)

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


test_basic_with_capture_message_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-ASfaDugIT60RnKtXL11x7yXKEn7WK",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "Atlantic Ocean",
        "chatcmpl-ASxkJuCwyegZk4W2awEhTKyCzstRr",
        24,
        2,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "Atlantic Ocean",
        "chatcmpl-913",
        46,
        3,
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_basic_with_capture_message_content_test_data,
)
def test_basic_with_capture_message_content(
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
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
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

    assert_stop_log_record(choice, content)

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
        "chatcmpl-ASfaENPdLwpR8Jo5gHFiIL9tuklHK",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "Atlantic Ocean",
        "chatcmpl-ASxkK7GCpziJMmUFp08jG5xb3Rr9K",
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "Atlantic Sea.",
        "chatcmpl-702",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize("provider_str,model,response_model,content,response_id,duration", test_stream_test_data)
def test_stream(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    duration,
    trace_exporter,
    metrics_reader,
    logs_exporter,
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice, expected_content=None)

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )


test_stream_all_the_client_options_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-Ab7br3ArYb5ZSjD5Z4ujJO3zlnmU6",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4o-mini",
        "South Atlantic Ocean.",
        "chatcmpl-Ab7bsWDRtmzRV9yhkTN8fEPJW0Z8r",
        24,
        4,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "Amalfis Sea",
        "chatcmpl-75",
        46,
        5,
        0.002600736916065216,
    ),
]


@pytest.mark.skipif(OPENAI_VERSION < (1, 35, 0), reason="service tier added in 1.35.0")
@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_stream_all_the_client_options_test_data,
)
def test_stream_all_the_client_options(
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
        seed=100,
        service_tier="default",
        response_format={"type": "text"},
        stream=True,
    )

    chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    expected_attrs = {
        GEN_AI_OPENAI_REQUEST_SEED: 100,
        GEN_AI_OPENAI_REQUEST_SERVICE_TIER: "default",
        GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT: "text",
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
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
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }
    if provider_str != "openai_provider_chat_completions":
        del expected_attrs[GEN_AI_OPENAI_RESPONSE_SERVICE_TIER]
    assert dict(span.attributes) == expected_attrs

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice, expected_content=None)

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
        "chatcmpl-ASfaFZizr7oebXDx3CgQuBnXD01Xp",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "Atlantic Ocean",
        "chatcmpl-ASxkLviKAQt414bmMPfpr2DfNYgKt",
        24,
        2,
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
    logs_exporter,
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice, expected_content=None)

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
def test_stream_with_include_usage_option_and_capture_message_content_integration(
    provider_str,
    model,
    response_model,
    trace_exporter,
    metrics_reader,
    logs_exporter,
    request,
):
    provider = request.getfixturevalue(provider_str)

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"},
    ):
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {"content": "Answer in up to 3 words: Which ocean contains the falkland islands?"}

    assert_stop_log_record(choice, content)

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


test_stream_with_tools_and_capture_message_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "",
        '{"order_id": "order_12345"}',
        "chatcmpl-ASfaGWIziiqvP4PoufIFi3K2p2caY",
        "tool_calls",
        "call_oEbEusT5nkkAiGgypUSDwK7k",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "",
        '{"order_id": "order_12345"}',
        "chatcmpl-ASxkOzfrRK9uiquPJ2AH90npzayEy",
        "tool_calls",
        "call_U0QYBadhpy4pBO6jYPm09KvZ",
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
        "chatcmpl-749",
        "stop",
        "ciao",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,completion_content,response_id,finish_reason,function_call_id,duration",
    test_stream_with_tools_and_capture_message_content_test_data,
)
def test_stream_with_tools_and_capture_message_content(
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
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
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
        assert_tool_call_log_record(
            choice, [ToolCall(function_call_id, "get_delivery_date", '{"order_id": "order_12345"}')]
        )
    else:
        assert_stop_log_record(choice, content)

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


test_stream_with_parallel_tools_and_capture_message_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "",
        json.dumps(""),
        "chatcmpl-ASfaJQVIzX3LUZllbc5hR0NnEzs0b",
        "tool_calls",
        0.006761051714420319,
    ),
    # Azure is not tested because gpt-4-32k does not support parallel tool calls
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        """<tool_call>
{"name": "get_weather", "arguments": {"location": "New York City, New York"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "London, United Kingdom"}}
</tool_call>""",
        """{"message": {"content":"<tool_call>
{"name": "get_weather", "arguments": {"location": "New York City, New York"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"location": "London, United Kingdom"}}
</tool_call>"}}""",
        "chatcmpl-563",
        "stop",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,completion_content,response_id,finish_reason,duration",
    test_stream_with_parallel_tools_and_capture_message_content_test_data,
)
def test_stream_with_parallel_tools_and_capture_message_content(
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
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
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

    if finish_reason == "tool_calls":
        assert_tool_call_log_record(
            choice,
            [
                ToolCall("call_9nzwliy6hCnTQuvkNdULIFkr", "get_weather", '{"location": "New York City"}'),
                ToolCall("call_3WjOCSgcSXK5YPOuP6GMwncg", "get_weather", '{"location": "London"}'),
            ],
        )
    else:
        assert_stop_log_record(choice, content)

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


test_tools_with_followup_and_capture_message_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        None,
        json.dumps(""),
        "tool_calls",
        0.007433261722326279,
    ),
    # Azure is not tested because gpt-4-32k does not support parallel tool calls
    # ollama does not return tool calls
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,completion_content,finish_reason,duration",
    test_tools_with_followup_and_capture_message_content_test_data,
)
def test_tools_with_followup_and_capture_message_content(
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
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
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

    first_response_message = first_response.choices[0].message
    if hasattr(first_response_message, "to_dict"):
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

    assert_tool_call_log_record(
        choice,
        [
            ToolCall(
                id=previous_message["tool_calls"][0]["id"],
                name="get_weather",
                arguments_json='{"location": "New York City"}',
            ),
            ToolCall(
                id=previous_message["tool_calls"][1]["id"], name="get_weather", arguments_json='{"location": "London"}'
            ),
        ],
    )

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

    assert_stop_log_record(choice, second_response.choices[0].message.content)

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
        "chatcmpl-ASfaN2cUXcaAFQ7uFS83Kuvc0iKdp",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "Atlantic Ocean",
        "chatcmpl-ASxkRs5B2H6Nyi9F1xV78yBVkiBbi",
        24,
        2,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falkland Islands belong to Argentina.",
        "chatcmpl-95",
        46,
        9,
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
    logs_exporter,
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

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice, expected_content=None)

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


test_async_basic_with_capture_message_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-ASfaOykfLmr5qdUddSbFIDNGReNRJ",
        24,
        4,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "Atlantic Ocean",
        "chatcmpl-ASxkTeNjqCBy25d2g18faeBtc66GG",
        24,
        2,
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falkland Islands are located in which ocean?",
        "chatcmpl-295",
        46,
        11,
        0.002600736916065216,
    ),
]


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,input_tokens,output_tokens,duration",
    test_async_basic_with_capture_message_content_test_data,
)
async def test_async_basic_with_capture_message_content(
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
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
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

    assert_stop_log_record(choice, content)

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
async def test_async_basic_with_capture_message_content_integration(
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
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
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

    assert_stop_log_record(choice, content)

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
        "chatcmpl-ASfaPgUOEfxcTVUSEfUINfcFkVQ8x",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "Atlantic Ocean",
        "chatcmpl-ASxkU7Rz5UnZqCWoV86xfgDVgc719",
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Southern Ocean.",
        "chatcmpl-232",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize("provider_str,model,response_model,content,response_id,duration", test_async_stream_test_data)
async def test_async_stream(
    provider_str,
    model,
    response_model,
    content,
    response_id,
    duration,
    trace_exporter,
    metrics_reader,
    logs_exporter,
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
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice, expected_content=None)

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response_model,
    }
    assert_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, min_data_point=duration
    )


test_async_stream_with_capture_message_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "South Atlantic Ocean.",
        "chatcmpl-ASfaQS4L6eLFljyu9h8zZRbp2hf5j",
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "Atlantic Ocean",
        "chatcmpl-ASyRrEoGtgeeLRtFd7mM1CePrJUac",
        0.002889830619096756,
    ),
    (
        "ollama_provider_chat_completions",
        "qwen2.5:0.5b",
        "qwen2.5:0.5b",
        "The Falkland Islands belong to Argentina.",
        "chatcmpl-465",
        0.002600736916065216,
    ),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_str,model,response_model,content,response_id,duration",
    test_async_stream_with_capture_message_content_test_data,
)
async def test_async_stream_with_capture_message_content(
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
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
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

    assert_stop_log_record(choice, content)

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
test_async_tools_with_capture_message_content_test_data = [
    (
        "openai_provider_chat_completions",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "chatcmpl-ASfaSKFvgQzcktnJhUZmtGmdoKwkW",
        "call_TZeQV35RVjT4iAuUXs85nmkt",
        "",
        140,
        19,
        0.006761051714420319,
    ),
    (
        "azure_provider_chat_completions",
        "unused",
        "gpt-4-32k",
        "chatcmpl-ASxkYrqUQVubUzovieQByym7sxgdm",
        "call_U0QYBadhpy4pBO6jYPm09KvZ",
        "",
        144,
        20,
        0.002889830619096756,
    ),
]


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_str,model,response_model,response_id,function_call_id,choice_content,input_tokens,output_tokens,duration",
    test_async_tools_with_capture_message_content_test_data,
)
async def test_async_tools_with_capture_message_content(
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
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
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

    assert_tool_call_log_record(
        choice, [ToolCall(function_call_id, "get_delivery_date", '{"order_id": "order_12345"}')]
    )

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
    ("openai_provider_chat_completions", 5),
    ("azure_provider_chat_completions", 5),
    (
        "ollama_provider_chat_completions",
        5,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize("provider_str,duration", test_without_model_parameter_test_data)
def test_without_model_parameter(
    provider_str,
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
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        "error.type": "TypeError",
        "server.address": provider.server_address,
        "server.port": provider.server_port,
    }
    assert_error_operation_duration_metric(
        provider, operation_duration_metric, attributes=attributes, data_point=duration, value_delta=5
    )


test_with_model_not_found_test_data = [
    (
        "openai_provider_chat_completions",
        "The model `not-found-model` does not exist or you do not have access to it.",
        0.00230291485786438,
    ),
    # We don't test with azure because the model parameter is ignored, so ends up successful.
    # This is verifiable by noticing the model parameter is always set to "unused"
    (
        "ollama_provider_chat_completions",
        'model "not-found-model" not found, try pulling it first',
        0.00230291485786438,
    ),
]


@pytest.mark.vcr()
@pytest.mark.parametrize("provider_str,exception,duration", test_with_model_not_found_test_data)
def test_with_model_not_found(
    provider_str,
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
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
    }

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        "gen_ai.request.model": "not-found-model",
        "error.type": "NotFoundError",
        SERVER_ADDRESS: provider.server_address,
        SERVER_PORT: provider.server_port,
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
    assert span.instrumentation_scope.schema_url == "https://opentelemetry.io/schemas/1.28.0"

    metrics_data = metrics_reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    for metrics in resource_metrics:
        for scope_metrics in metrics.scope_metrics:
            assert scope_metrics.schema_url == "https://opentelemetry.io/schemas/1.28.0"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_json: str


def assert_stop_log_record(log_record: LogRecord, expected_content: str, expected_index=0):
    assert log_record.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}
    assert log_record.body["index"] == expected_index
    assert log_record.body["finish_reason"] == "stop"
    message = log_record.body["message"]
    if expected_content is None:
        assert "content" not in message
    else:
        assert message["content"] == expected_content


def assert_tool_call_log_record(log_record: LogRecord, expected_tool_calls: List[ToolCall], expected_index=0):
    assert log_record.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.choice"}
    assert log_record.body["index"] == expected_index
    assert log_record.body["finish_reason"] == "tool_calls"
    message = log_record.body["message"]
    assert_tool_calls(message["tool_calls"], expected_tool_calls)


def assert_tool_call_event(event: Event, expected_tool_calls: List[ToolCall]):
    assert event.name == "gen_ai.content.completion"
    # The 'gen_ai.completion' attribute is a JSON string, so parse it first.
    gen_ai_completions = json.loads(event.attributes["gen_ai.completion"])

    gen_ai_completion = gen_ai_completions[0]
    assert gen_ai_completion["role"] == "assistant"
    assert gen_ai_completion["content"] == ""
    assert_tool_calls(gen_ai_completion["tool_calls"], expected_tool_calls)


def assert_tool_calls(tool_calls, expected_tool_calls: List[ToolCall]):
    for i, tool_call in enumerate(tool_calls):
        expected_call = expected_tool_calls[i]
        args = tool_call["function"]["arguments"]
        # The function arguments are also a string, which has different whitespace
        # in Azure. Assert in a whitespace agnostic way first.
        assert json.dumps(json.loads(args), sort_keys=True) == expected_call.arguments_json

        assert tool_call == {
            "id": expected_call.id,
            "type": "function",
            "function": {"name": expected_call.name, "arguments": args},
        }, f"Unexpected tool_call at index {i}: {tool_call} != {expected_call}"
