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
import os
import re
from dataclasses import dataclass
from typing import List, Optional
from unittest import mock

import openai
import pytest
from opentelemetry._events import Event
from opentelemetry._logs import LogRecord
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPENAI_REQUEST_SERVICE_TIER,
    GEN_AI_OPENAI_RESPONSE_SERVICE_TIER,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_TYPE,
    GEN_AI_REQUEST_CHOICE_COUNT,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_SEED,
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
    address_and_port,
    assert_error_operation_duration_metric,
    assert_operation_duration_metric,
    assert_token_usage_metric,
    get_integration_async_client,
    get_integration_client,
)
from .utils import MOCK_POSITIVE_FLOAT, get_sorted_metrics, logrecords_from_logs

OPENAI_VERSION = tuple([int(x) for x in openai.version.VERSION.split(".")])
TEST_CHAT_MODEL = "gpt-4o-mini"
TEST_CHAT_RESPONSE_MODEL = "gpt-4o-mini-2024-07-18"
TEST_CHAT_INPUT = "Answer in up to 3 words: Which ocean contains Bouvet Island?"


@pytest.mark.vcr()
def test_chat(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages)

    assert chat_completion.choices[0].message.content == "Atlantic Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuGVpfQzbsboUTm9uUCSEUWwEbU",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: 22,
        GEN_AI_USAGE_OUTPUT_TOKENS: 3,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: "gpt-4o-mini-2024-07-18",
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.vcr()
def test_chat_n_1(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, n=1)

    assert chat_completion.choices[0].message.content == "Atlantic Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert GEN_AI_REQUEST_CHOICE_COUNT not in span.attributes


@pytest.mark.skipif(OPENAI_VERSION < (1, 8, 0), reason="LegacyAPIResponse available")
@pytest.mark.vcr()
def test_chat_with_raw_response(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = client.chat.completions.with_raw_response.create(model=TEST_CHAT_MODEL, messages=messages)

    assert chat_completion.choices[0].message.content == "South Atlantic Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-BDDnDacM4nUxi3Qsplkrewf7L7Y10",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: 22,
        GEN_AI_USAGE_OUTPUT_TOKENS: 5,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: "gpt-4o-mini-2024-07-18",
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.vcr()
def test_chat_with_developer_role_message(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

    messages = [
        {
            "role": "developer",
            "content": "You are a friendly assistant",
        },
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        },
    ]

    chat_completion = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages)

    assert chat_completion.choices[0].message.content == "Atlantic Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-B6vdHtqgT6rj4cj7itn9bNlaUlqHg",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: 31,
        GEN_AI_USAGE_OUTPUT_TOKENS: 4,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 3
    log_records = logrecords_from_logs(logs)
    system_message, user_message, choice = log_records
    assert dict(system_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.system.message"}
    assert dict(system_message.body) == {"role": "developer"}

    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: "gpt-4o-mini-2024-07-18",
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.skipif(OPENAI_VERSION < (1, 35, 0), reason="service tier added in 1.35.0")
@pytest.mark.vcr()
def test_chat_all_the_client_options(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    params = {
        "model": TEST_CHAT_MODEL,
        "messages": messages,
        "frequency_penalty": 0,
        "max_completion_tokens": 100,
        "presence_penalty": 0,
        "temperature": 1,
        "top_p": 1,
        "stop": "foo",
        "seed": 100,
        "service_tier": "default",
        "response_format": {"type": "text"},
    }
    chat_completion = client.chat.completions.create(**params)

    assert chat_completion.choices[0].message.content == "Southern Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    expected_attrs = {
        GEN_AI_REQUEST_SEED: 100,
        GEN_AI_OPENAI_REQUEST_SERVICE_TIER: "default",
        GEN_AI_OUTPUT_TYPE: "text",
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_FREQUENCY_PENALTY: 0,
        GEN_AI_REQUEST_MAX_TOKENS: 100,
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_REQUEST_PRESENCE_PENALTY: 0,
        GEN_AI_REQUEST_STOP_SEQUENCES: ("foo",),
        GEN_AI_REQUEST_TEMPERATURE: 1,
        GEN_AI_REQUEST_TOP_P: 1,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhvFSrCe0B1E6Prdwn9U7V2Lq8XH",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: 22,
        GEN_AI_USAGE_OUTPUT_TOKENS: 3,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }
    assert dict(span.attributes) == expected_attrs

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.skipif(OPENAI_VERSION < (1, 35, 0), reason="service tier added in 1.35.0")
@pytest.mark.vcr()
def test_chat_all_the_client_options_not_given(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    params = {
        "model": TEST_CHAT_MODEL,
        "messages": messages,
        "frequency_penalty": openai.NOT_GIVEN,
        "max_completion_tokens": openai.NOT_GIVEN,
        "presence_penalty": openai.NOT_GIVEN,
        "temperature": openai.NOT_GIVEN,
        "top_p": openai.NOT_GIVEN,
        "stop": openai.NOT_GIVEN,
        "seed": openai.NOT_GIVEN,
        "service_tier": openai.NOT_GIVEN,
        "response_format": openai.NOT_GIVEN,
    }
    chat_completion = client.chat.completions.create(**params)

    assert chat_completion.choices[0].message.content == "Atlantic Ocean"

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    expected_attrs = {
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-BCOdmGkOZ511LwlA800bJkFWf528Z",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: 22,
        GEN_AI_USAGE_OUTPUT_TOKENS: 3,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }
    assert dict(span.attributes) == expected_attrs

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.vcr()
def test_chat_multiple_choices_with_capture_message_content(
    default_openai_env, trace_exporter, metrics_reader, logs_exporter
):
    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, n=2)

    content = "Atlantic Ocean."
    assert chat_completion.choices[0].message.content == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_CHOICE_COUNT: 2,
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuHpVEbcYGlsFuHOP60MtU4tIq9",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop", "stop"),
        GEN_AI_USAGE_INPUT_TOKENS: 22,
        GEN_AI_USAGE_OUTPUT_TOKENS: 6,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 3
    log_records = logrecords_from_logs(logs)
    user_message, choice, second_choice = log_records
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {"content": TEST_CHAT_INPUT}

    assert_stop_log_record(choice, content)
    assert_stop_log_record(second_choice, "Southern Ocean.", 1)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.vcr()
def test_chat_function_calling_with_tools(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

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

    response = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, tools=tools)
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_delivery_date"
    # FIXME: add to test data
    assert json.loads(tool_call.function.arguments) == {"order_id": "order_12345"}

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuIeTQU1AlqGqx3cfvtbNyJ2Q8p",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
        GEN_AI_USAGE_INPUT_TOKENS: 140,
        GEN_AI_USAGE_OUTPUT_TOKENS: 19,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
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
        choice, [ToolCall("call_BAohHzhtwXBSM13jKADbwgQH", "get_delivery_date", '{"order_id": "order_12345"}')]
    )

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.vcr()
def test_chat_tools_with_capture_message_content(default_openai_env, trace_exporter, logs_exporter, metrics_reader):
    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    client = openai.OpenAI()

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

    response = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, tools=tools)
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_delivery_date"
    assert json.loads(tool_call.function.arguments) == {"order_id": "order_12345"}

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuJxYuidCW2KvkwBy6VMnWtdiwb",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
        GEN_AI_USAGE_INPUT_TOKENS: 140,
        GEN_AI_USAGE_OUTPUT_TOKENS: 19,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
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
        choice, [ToolCall("call_TD1k1LOj7QC0uQPRihIY9Bml", "get_delivery_date", '{"order_id": "order_12345"}')]
    )

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.integration
def test_chat_tools_with_capture_message_content_integration(trace_exporter, logs_exporter, metrics_reader):
    client = get_integration_client()
    model = os.getenv("TEST_CHAT_MODEL", TEST_CHAT_MODEL)

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

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response.id,
        GEN_AI_RESPONSE_MODEL: response.model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
        GEN_AI_USAGE_INPUT_TOKENS: response.usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: response.usage.completion_tokens,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
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
        GEN_AI_RESPONSE_MODEL: response.model,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=MOCK_POSITIVE_FLOAT
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


def test_chat_connection_error(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.Client(base_url="http://localhost:9999/v5", api_key="not-read", max_retries=1)
    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    with pytest.raises(Exception):
        client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages)

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
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
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        ERROR_TYPE: "APIConnectionError",
    }
    assert_error_operation_duration_metric(
        "chat",
        operation_duration_metric,
        attributes=attributes,
        data_point=1.026234219999992,
        value_delta=1.0,
    )


@pytest.mark.integration
def test_chat_with_capture_message_content_integration(trace_exporter, logs_exporter, metrics_reader):
    model = os.getenv("TEST_CHAT_MODEL", TEST_CHAT_MODEL)

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"},
    ):
        OpenAIInstrumentor().instrument()

    client = get_integration_client()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
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

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response.id,
        GEN_AI_RESPONSE_MODEL: response.model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: response.usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: response.usage.completion_tokens,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {"content": TEST_CHAT_INPUT}

    assert_stop_log_record(choice, content)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response.model,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=MOCK_POSITIVE_FLOAT
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.vcr()
def test_chat_with_capture_message_content(default_openai_env, trace_exporter, logs_exporter, metrics_reader):
    client = openai.OpenAI()

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages)

    content = "South Atlantic Ocean."
    assert chat_completion.choices[0].message.content == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuKQOLh8rjzshDoq35O7wceMSEK",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: 22,
        GEN_AI_USAGE_OUTPUT_TOKENS: 4,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {"content": TEST_CHAT_INPUT}

    assert_stop_log_record(choice, content)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.vcr()
def test_chat_stream(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, stream=True)

    chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == "South Atlantic Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuL5q147VH6OYeahA32U4bM3p1o",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )


@pytest.mark.vcr()
def test_chat_stream_with_context_manager(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    # Use a context manager for the streaming response
    with client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, stream=True) as chat_completion:
        chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
        assert "".join(chunks) == "South Atlantic Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-BOja7e365tj5upRjLFinadEB8ZoDL",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )


@pytest.mark.skipif(OPENAI_VERSION < (1, 8, 0), reason="LegacyAPIResponse available")
@pytest.mark.vcr()
def test_chat_stream_with_raw_response(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    raw_response = client.chat.completions.with_raw_response.create(
        model=TEST_CHAT_MODEL, messages=messages, stream=True
    )

    # Explicit parse of the raw response
    chat_completion = raw_response.parse()

    chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == "Atlantic Ocean"

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-BDDnEHqYLBd36X8hHNTQfPKx4KMJT",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )


@pytest.mark.skipif(OPENAI_VERSION < (1, 35, 0), reason="service tier added in 1.35.0")
@pytest.mark.vcr()
def test_chat_stream_all_the_client_options(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    params = {
        "model": TEST_CHAT_MODEL,
        "messages": messages,
        "frequency_penalty": 0,
        "max_tokens": 100,
        "presence_penalty": 0,
        "temperature": 1,
        "top_p": 1,
        "stop": "foo",
        "seed": 100,
        "service_tier": "default",
        "response_format": {"type": "text"},
        "stream": True,
    }
    chat_completion = client.chat.completions.create(**params)

    chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == "Southern Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    expected_attrs = {
        GEN_AI_REQUEST_SEED: 100,
        GEN_AI_OUTPUT_TYPE: "text",
        GEN_AI_OPENAI_REQUEST_SERVICE_TIER: "default",
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_FREQUENCY_PENALTY: 0,
        GEN_AI_REQUEST_MAX_TOKENS: 100,
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_REQUEST_PRESENCE_PENALTY: 0,
        GEN_AI_REQUEST_STOP_SEQUENCES: ("foo",),
        GEN_AI_REQUEST_TEMPERATURE: 1,
        GEN_AI_REQUEST_TOP_P: 1,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuLKifS5fLvJL3oXhlJesoo5goa",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }
    assert dict(span.attributes) == expected_attrs

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )


@pytest.mark.skipif(OPENAI_VERSION < (1, 26, 0), reason="stream_options added in 1.26.0")
@pytest.mark.vcr()
def test_chat_stream_with_include_usage_option(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = client.chat.completions.create(
        model=TEST_CHAT_MODEL, messages=messages, stream=True, stream_options={"include_usage": True}
    )

    chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == "Southern Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuMhGf8Genpm2uKosEpBtvtQgco",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: 22,
        GEN_AI_USAGE_OUTPUT_TOKENS: 3,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.skipif(OPENAI_VERSION < (1, 26, 0), reason="stream_options added in 1.26.0")
@pytest.mark.integration
def test_chat_stream_with_include_usage_option_and_capture_message_content_integration(
    default_openai_env, trace_exporter, logs_exporter, metrics_reader
):
    model = os.getenv("TEST_CHAT_MODEL", TEST_CHAT_MODEL)

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"},
    ):
        OpenAIInstrumentor().instrument()

    client = get_integration_client()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
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

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: chunks[0].id if port != 11434 else "",  # ollama doesn't return the id
        GEN_AI_RESPONSE_MODEL: response.model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: usage.completion_tokens,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {"content": TEST_CHAT_INPUT}

    assert_stop_log_record(choice, content)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response.model,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=MOCK_POSITIVE_FLOAT
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=usage.prompt_tokens,
        output_data_point=usage.completion_tokens,
    )


@pytest.mark.vcr()
def test_chat_stream_with_tools_and_capture_message_content(
    default_openai_env, trace_exporter, logs_exporter, metrics_reader
):
    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    client = openai.OpenAI()

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

    chat_completion = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, tools=tools, stream=True)

    chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == ""

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuNDvVfpeTo6dHUzXlhuCKVNMyp",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
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

    assert_tool_call_log_record(
        choice, [ToolCall("call_BiJxky21FTzGOu7GDBGK7SHq", "get_delivery_date", '{"order_id": "order_12345"}')]
    )

    span_ctx = span.get_span_context()
    assert choice.trace_id == span_ctx.trace_id
    assert choice.span_id == span_ctx.span_id
    assert choice.trace_flags == span_ctx.trace_flags

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )


@pytest.mark.vcr()
def test_chat_stream_with_parallel_tools_and_capture_message_content(
    default_openai_env, trace_exporter, logs_exporter, metrics_reader
):
    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    client = openai.OpenAI()

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

    chat_completion = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, tools=tools, stream=True)

    chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == ""

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuOt4HonDIz3Pbi86KHAwnzhJTZ",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 3
    log_records = logrecords_from_logs(logs)
    system_message, user_message, choice = log_records
    assert system_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.system.message"}
    assert system_message.body == {"content": "You are a helpful assistant providing weather updates."}
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {"content": "What is the weather in New York City and London?"}

    assert_tool_call_log_record(
        choice,
        [
            ToolCall("call_vbjItaL3xe3uYPY1PIVhmBcs", "get_weather", '{"location": "New York City"}'),
            ToolCall("call_q2Px0dkOQv47VpcCF50xZsap", "get_weather", '{"location": "London"}'),
        ],
    )

    span_ctx = span.get_span_context()
    assert choice.trace_id == span_ctx.trace_id
    assert choice.span_id == span_ctx.span_id
    assert choice.trace_flags == span_ctx.trace_flags

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )


@pytest.mark.vcr()
def test_chat_tools_with_followup_and_capture_message_content(
    default_openai_env, trace_exporter, metrics_reader, logs_exporter
):
    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    client = openai.OpenAI()

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

    first_response = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, tools=tools)

    assert first_response.choices[0].message.content is None

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

    second_response = client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages + followup_messages)

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 2

    first_span, second_span = spans
    assert first_span.name == f"chat {TEST_CHAT_MODEL}"
    assert first_span.kind == SpanKind.CLIENT
    assert first_span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(first_span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: first_response.id,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
        GEN_AI_USAGE_INPUT_TOKENS: first_response.usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: first_response.usage.completion_tokens,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    assert second_span.name == f"chat {TEST_CHAT_MODEL}"
    assert second_span.kind == SpanKind.CLIENT
    assert second_span.status.status_code == StatusCode.UNSET

    assert dict(second_span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: second_response.id,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: second_response.usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: second_response.usage.completion_tokens,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
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
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.007433261722326279, count=2
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=[first_response.usage.prompt_tokens, second_response.usage.prompt_tokens],
        output_data_point=[first_response.usage.completion_tokens, second_response.usage.completion_tokens],
        count=2,
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_async(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.AsyncOpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = await client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages)

    assert chat_completion.choices[0].message.content == "Atlantic Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuGVpfQzbsboUTm9uUCSEUWwEbU",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: 22,
        GEN_AI_USAGE_OUTPUT_TOKENS: 3,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.skipif(OPENAI_VERSION < (1, 8, 0), reason="LegacyAPIResponse available")
@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_async_with_raw_response(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.AsyncOpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = await client.chat.completions.with_raw_response.create(model=TEST_CHAT_MODEL, messages=messages)

    assert chat_completion.choices[0].message.content == "South Atlantic Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-BDDnDacM4nUxi3Qsplkrewf7L7Y10",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: 22,
        GEN_AI_USAGE_OUTPUT_TOKENS: 5,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_async_with_capture_message_content(
    default_openai_env, trace_exporter, metrics_reader, logs_exporter
):
    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    client = openai.AsyncOpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = await client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages)

    content = "South Atlantic Ocean."
    assert chat_completion.choices[0].message.content == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuKQOLh8rjzshDoq35O7wceMSEK",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: 22,
        GEN_AI_USAGE_OUTPUT_TOKENS: 4,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {"content": TEST_CHAT_INPUT}

    assert_stop_log_record(choice, content)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_async_with_capture_message_content_integration(trace_exporter, logs_exporter, metrics_reader):
    model = os.getenv("TEST_CHAT_MODEL", TEST_CHAT_MODEL)

    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    client = get_integration_async_client()

    response = await client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    assert content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: response.id,
        GEN_AI_RESPONSE_MODEL: response.model,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        GEN_AI_USAGE_INPUT_TOKENS: response.usage.prompt_tokens,
        GEN_AI_USAGE_OUTPUT_TOKENS: response.usage.completion_tokens,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert user_message.attributes == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert user_message.body == {"content": TEST_CHAT_INPUT}

    assert_stop_log_record(choice, content)

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_RESPONSE_MODEL: response.model,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=MOCK_POSITIVE_FLOAT
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=response.usage.prompt_tokens,
        output_data_point=response.usage.completion_tokens,
    )


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_chat_async_stream(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.AsyncOpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = await client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, stream=True)

    chunks = [chunk.choices[0].delta.content or "" async for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == "South Atlantic Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuL5q147VH6OYeahA32U4bM3p1o",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_chat_async_stream_with_context_manager(
    default_openai_env, trace_exporter, metrics_reader, logs_exporter
):
    client = openai.AsyncOpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    # Use a context manager for the asynchronous streaming response
    async with await client.chat.completions.create(
        model=TEST_CHAT_MODEL, messages=messages, stream=True
    ) as chat_completion:
        chunks = [chunk.choices[0].delta.content or "" async for chunk in chat_completion if chunk.choices]
        assert "".join(chunks) == "South Atlantic Ocean."

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-BOja7e365tj5upRjLFinadEB8ZoDL",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )


@pytest.mark.skipif(OPENAI_VERSION < (1, 8, 0), reason="LegacyAPIResponse available")
@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_chat_async_stream_with_raw_response(default_openai_env, trace_exporter, metrics_reader, logs_exporter):
    client = openai.AsyncOpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    raw_response = await client.chat.completions.with_raw_response.create(
        model=TEST_CHAT_MODEL, messages=messages, stream=True
    )

    # Explicit parse of the raw response
    chat_completion = raw_response.parse()

    chunks = [chunk.choices[0].delta.content or "" async for chunk in chat_completion if chunk.choices]
    assert "".join(chunks) == "Atlantic Ocean"

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPENAI_RESPONSE_SERVICE_TIER: "default",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-BDDnEHqYLBd36X8hHNTQfPKx4KMJT",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {}

    assert_stop_log_record(choice)

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_chat_async_stream_with_capture_message_content(
    default_openai_env, trace_exporter, metrics_reader, logs_exporter
):
    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    client = openai.AsyncOpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    chat_completion = await client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, stream=True)

    chunks = [chunk.choices[0].delta.content or "" async for chunk in chat_completion if chunk.choices]
    content = "South Atlantic Ocean."
    assert "".join(chunks) == content

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuREFCucwYqmosV554sWlHEdQmW",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    logs = logs_exporter.get_finished_logs()
    assert len(logs) == 2
    log_records = logrecords_from_logs(logs)
    user_message, choice = log_records
    assert dict(user_message.attributes) == {"gen_ai.system": "openai", "event.name": "gen_ai.user.message"}
    assert dict(user_message.body) == {"content": TEST_CHAT_INPUT}

    assert_stop_log_record(choice, content)

    span_ctx = span.get_span_context()
    assert choice.trace_id == span_ctx.trace_id
    assert choice.span_id == span_ctx.span_id
    assert choice.trace_flags == span_ctx.trace_flags

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_chat_async_tools_with_capture_message_content(
    default_openai_env, trace_exporter, metrics_reader, logs_exporter
):
    # Redo the instrumentation dance to be affected by the environment variable
    OpenAIInstrumentor().uninstrument()
    with mock.patch.dict("os.environ", {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true"}):
        OpenAIInstrumentor().instrument()

    client = openai.AsyncOpenAI()

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

    response = await client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages, tools=tools)
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_delivery_date"
    assert json.loads(tool_call.function.arguments) == {"order_id": "order_12345"}

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {TEST_CHAT_MODEL}"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_SYSTEM: "openai",
        GEN_AI_RESPONSE_ID: "chatcmpl-AfhuJxYuidCW2KvkwBy6VMnWtdiwb",
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
        GEN_AI_USAGE_INPUT_TOKENS: 140,
        GEN_AI_USAGE_OUTPUT_TOKENS: 19,
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
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
        choice, [ToolCall("call_TD1k1LOj7QC0uQPRihIY9Bml", "get_delivery_date", '{"order_id": "order_12345"}')]
    )

    operation_duration_metric, token_usage_metric = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: TEST_CHAT_MODEL,
        GEN_AI_RESPONSE_MODEL: TEST_CHAT_RESPONSE_MODEL,
    }
    assert_operation_duration_metric(
        client, "chat", operation_duration_metric, attributes=attributes, min_data_point=0.006761051714420319
    )
    assert_token_usage_metric(
        client,
        "chat",
        token_usage_metric,
        attributes=attributes,
        input_data_point=span.attributes[GEN_AI_USAGE_INPUT_TOKENS],
        output_data_point=span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS],
    )


@pytest.mark.vcr()
def test_chat_without_model_parameter(default_openai_env, trace_exporter, metrics_reader):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
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

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        ERROR_TYPE: "TypeError",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_SYSTEM: "openai",
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        "error.type": "TypeError",
        "server.address": address,
        "server.port": port,
    }
    assert_error_operation_duration_metric(
        "chat", operation_duration_metric, attributes=attributes, data_point=5, value_delta=5
    )


@pytest.mark.vcr()
def test_chat_with_model_not_found(default_openai_env, trace_exporter, metrics_reader):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    exception = "The model `not-found-TEST_CHAT_MODEL` does not exist or you do not have access to it."
    with pytest.raises(openai.NotFoundError, match="Error code: 404.*" + re.escape(exception)):
        client.chat.completions.create(model="not-found-TEST_CHAT_MODEL", messages=messages)

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "chat not-found-TEST_CHAT_MODEL"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.ERROR

    address, port = address_and_port(client)
    assert dict(span.attributes) == {
        ERROR_TYPE: "NotFoundError",
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: "not-found-TEST_CHAT_MODEL",
        GEN_AI_SYSTEM: "openai",
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }

    (operation_duration_metric,) = get_sorted_metrics(metrics_reader)
    attributes = {
        GEN_AI_REQUEST_MODEL: "not-found-TEST_CHAT_MODEL",
        "error.type": "NotFoundError",
        SERVER_ADDRESS: address,
        SERVER_PORT: port,
    }
    assert_error_operation_duration_metric(
        "chat", operation_duration_metric, attributes=attributes, data_point=0.00230291485786438
    )


@pytest.mark.vcr()
def test_chat_exported_schema_version(default_openai_env, trace_exporter, metrics_reader):
    client = openai.OpenAI()

    messages = [
        {
            "role": "user",
            "content": TEST_CHAT_INPUT,
        }
    ]

    client.chat.completions.create(model=TEST_CHAT_MODEL, messages=messages)

    spans = trace_exporter.get_finished_spans()
    (span,) = spans
    assert span.instrumentation_scope.schema_url == "https://opentelemetry.io/schemas/1.31.0"

    metrics_data = metrics_reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics

    for metrics in resource_metrics:
        for scope_metrics in metrics.scope_metrics:
            assert scope_metrics.schema_url == "https://opentelemetry.io/schemas/1.31.0"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_json: str


def assert_stop_log_record(log_record: LogRecord, expected_content: Optional[str] = None, expected_index=0):
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
