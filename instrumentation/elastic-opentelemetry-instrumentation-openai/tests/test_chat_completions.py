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
from unittest import IsolatedAsyncioTestCase, mock

import openai
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.test.test_base import TestBase
from opentelemetry.metrics import Histogram
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
from vcr.unittest import VCRMixin

# Use the same model for tools as for chat completion
OPENAI_TOOL_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = "test_openai_api_key"
OPENAI_ORG_ID = "test_openai_org_key"
OPENAI_PROJECT_ID = "test_openai_project_id"

LOCAL_MODEL = "sam4096/qwen2tools:0.5b"


class OpenaiMixin(VCRMixin):
    def _get_vcr_kwargs(self, **kwargs):
        """
        This scrubs sensitive data when in recording mode.
        """
        return {
            "filter_headers": [
                ("authorization", "Bearer " + OPENAI_API_KEY),
                ("openai-organization", OPENAI_ORG_ID),
                ("openai-project", OPENAI_PROJECT_ID),
                ("cookie", None),
            ],
            "before_record_response": self.scrub_response_headers,
        }

    @staticmethod
    def scrub_response_headers(response):
        """
        This scrubs sensitive response headers. Note they are case-sensitive!
        """
        response["headers"]["openai-organization"] = OPENAI_ORG_ID
        response["headers"]["Set-Cookie"] = "test_set_cookie"
        return response

    @classmethod
    def setup_client(cls):
        # Control the arguments
        return openai.Client(
            api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
            organization=os.getenv("OPENAI_ORG_ID", OPENAI_ORG_ID),
            project=os.getenv("OPENAI_PROJECT_ID", OPENAI_PROJECT_ID),
            max_retries=1,
        )

    @classmethod
    def setUpClass(cls):
        cls.client = cls.setup_client()

    def setUp(self):
        super().setUp()
        OpenAIInstrumentor().instrument()

    def tearDown(self):
        super().tearDown()
        OpenAIInstrumentor().uninstrument()

    def assertOperationDurationMetric(self, metric: Histogram):
        self.assertEqual(metric.name, "gen_ai.client.operation.duration")
        self.assert_metric_expected(
            metric,
            [
                self.create_histogram_data_point(
                    count=1,
                    sum_data_point=0.006543334107846022,
                    max_data_point=0.006543334107846022,
                    min_data_point=0.006543334107846022,
                    attributes={
                        "gen_ai.operation.name": "chat",
                        "gen_ai.request.model": OPENAI_TOOL_MODEL,
                        "gen_ai.response.model": f"{OPENAI_TOOL_MODEL}-2024-07-18",
                        "gen_ai.system": "openai",
                        "server.address": "api.openai.com",
                        "server.port": 443,
                    },
                ),
            ],
            est_value_delta=0.1,
        )

    def assertErrorOperationDurationMetric(self, metric: Histogram, attributes: dict, data_point: float = None):
        self.assertEqual(metric.name, "gen_ai.client.operation.duration")
        default_attributes = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": OPENAI_TOOL_MODEL,
            "gen_ai.system": "openai",
            "error.type": "APIConnectionError",
            "server.address": "localhost",
            "server.port": 9999,
        }
        if data_point is None:
            data_point = 0.8643839359283447
        self.assert_metric_expected(
            metric,
            [
                self.create_histogram_data_point(
                    count=1,
                    sum_data_point=data_point,
                    max_data_point=data_point,
                    min_data_point=data_point,
                    attributes={**default_attributes, **attributes},
                ),
            ],
            est_value_delta=0.5,
        )

    def assertTokenUsageMetric(self, metric: Histogram, input_data_point=24, output_data_point=4):
        self.assertEqual(metric.name, "gen_ai.client.token.usage")
        self.assert_metric_expected(
            metric,
            [
                self.create_histogram_data_point(
                    count=1,
                    sum_data_point=input_data_point,
                    max_data_point=input_data_point,
                    min_data_point=input_data_point,
                    attributes={
                        "gen_ai.operation.name": "chat",
                        "gen_ai.request.model": OPENAI_TOOL_MODEL,
                        "gen_ai.response.model": f"{OPENAI_TOOL_MODEL}-2024-07-18",
                        "gen_ai.system": "openai",
                        "server.address": "api.openai.com",
                        "server.port": 443,
                        "gen_ai.token.type": "input",
                    },
                ),
                self.create_histogram_data_point(
                    count=1,
                    sum_data_point=output_data_point,
                    max_data_point=output_data_point,
                    min_data_point=output_data_point,
                    attributes={
                        "gen_ai.operation.name": "chat",
                        "gen_ai.request.model": OPENAI_TOOL_MODEL,
                        "gen_ai.response.model": f"{OPENAI_TOOL_MODEL}-2024-07-18",
                        "gen_ai.system": "openai",
                        "server.address": "api.openai.com",
                        "server.port": 443,
                        "gen_ai.token.type": "output",
                    },
                ),
            ],
        )


class TestChatCompletions(OpenaiMixin, TestBase):
    def test_basic(self):
        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        chat_completion = self.client.chat.completions.create(model=OPENAI_TOOL_MODEL, messages=messages)

        self.assertEqual(chat_completion.choices[0].message.content, "South Atlantic Ocean.")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-A9CSutUkLCxwZIXuXRXlgEJUCMnlT",
                GEN_AI_RESPONSE_MODEL: OPENAI_TOOL_MODEL + "-2024-07-18",  # Note it is more specific than request!
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                GEN_AI_USAGE_INPUT_TOKENS: 24,
                GEN_AI_USAGE_OUTPUT_TOKENS: 4,
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )
        self.assertEqual(span.events, ())

        operation_duration_metric, token_usage_metric = self.get_sorted_metrics()
        self.assertOperationDurationMetric(operation_duration_metric)
        self.assertTokenUsageMetric(token_usage_metric)

    def test_all_the_client_options(self):
        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        chat_completion = self.client.chat.completions.create(
            model=OPENAI_TOOL_MODEL,
            messages=messages,
            frequency_penalty=0,
            max_completion_tokens=100,
            presence_penalty=0,
            temperature=1,
            top_p=1,
            stop="foo",
        )

        self.assertEqual(chat_completion.choices[0].message.content, "South Atlantic Ocean.")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_FREQUENCY_PENALTY: 0,
                GEN_AI_REQUEST_MAX_TOKENS: 100,
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_REQUEST_PRESENCE_PENALTY: 0,
                GEN_AI_REQUEST_STOP_SEQUENCES: ("foo",),
                GEN_AI_REQUEST_TEMPERATURE: 1,
                GEN_AI_REQUEST_TOP_P: 1,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-ADUdg61PwWqn3FPn4VNkz4vwMkS62",
                GEN_AI_RESPONSE_MODEL: OPENAI_TOOL_MODEL + "-2024-07-18",  # Note it is more specific than request!,
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                GEN_AI_USAGE_INPUT_TOKENS: 24,
                GEN_AI_USAGE_OUTPUT_TOKENS: 4,
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )
        self.assertEqual(span.events, ())

        operation_duration_metric, token_usage_metric = self.get_sorted_metrics()
        self.assertOperationDurationMetric(operation_duration_metric)
        self.assertTokenUsageMetric(token_usage_metric)

    def test_function_calling_with_tools(self):
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

        response = self.client.chat.completions.create(model=OPENAI_TOOL_MODEL, messages=messages, tools=tools)
        tool_call = response.choices[0].message.tool_calls[0]
        self.assertEqual(tool_call.function.name, "get_delivery_date")
        self.assertEqual(json.loads(tool_call.function.arguments), {"order_id": "order_12345"})

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-A9CSwJdb2481bxsjIiuD8yIBOAfql",
                GEN_AI_RESPONSE_MODEL: OPENAI_TOOL_MODEL + "-2024-07-18",  # Note it is more specific than request!
                GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
                GEN_AI_USAGE_INPUT_TOKENS: 140,
                GEN_AI_USAGE_OUTPUT_TOKENS: 19,
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )
        self.assertEqual(span.events, ())

        operation_duration_metric, token_usage_metric = self.get_sorted_metrics()
        self.assertOperationDurationMetric(operation_duration_metric)
        self.assertTokenUsageMetric(token_usage_metric, input_data_point=140, output_data_point=19)

    def test_tools_with_capture_content(self):
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

        response = self.client.chat.completions.create(model=OPENAI_TOOL_MODEL, messages=messages, tools=tools)
        tool_call = response.choices[0].message.tool_calls[0]
        self.assertEqual(tool_call.function.name, "get_delivery_date")
        self.assertEqual(json.loads(tool_call.function.arguments), {"order_id": "order_12345"})

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-A9CSzz613rRslGZpG79Js1deMz98G",
                GEN_AI_RESPONSE_MODEL: OPENAI_TOOL_MODEL + "-2024-07-18",  # Note it is more specific than request!
                GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
                GEN_AI_USAGE_INPUT_TOKENS: 140,
                GEN_AI_USAGE_OUTPUT_TOKENS: 19,
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )

        self.assertEqual(len(span.events), 2)
        prompt_event, completion_event = span.events
        self.assertEqual(prompt_event.name, "gen_ai.content.prompt")
        self.assertEqual(dict(prompt_event.attributes), {"gen_ai.prompt": json.dumps(messages)})
        self.assertEqual(completion_event.name, "gen_ai.content.completion")
        self.assertEqual(
            dict(completion_event.attributes),
            {"gen_ai.completion": '[{"role": "assistant", "content": {"order_id": "order_12345"}}]'},
        )

    def test_connection_error(self):
        client = openai.Client(base_url="http://localhost:9999/v5", api_key="unused", max_retries=1)
        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        self.assertRaises(Exception, lambda: client.chat.completions.create(model=OPENAI_TOOL_MODEL, messages=messages))

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                ERROR_TYPE: "APIConnectionError",
                SERVER_ADDRESS: "localhost",
                SERVER_PORT: 9999,
            },
        )
        self.assertEqual(span.events, ())

        (operation_duration_metric,) = self.get_sorted_metrics()
        self.assertErrorOperationDurationMetric(operation_duration_metric, {"error.type": "APIConnectionError"})

    def test_local(self):
        client = openai.Client(base_url="http://localhost:11434/v1", api_key="unused", max_retries=1)

        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        chat_completion = client.chat.completions.create(model=LOCAL_MODEL, messages=messages)

        self.assertEqual(
            chat_completion.choices[0].message.content, "The South Atlantic Ocean contains the Falklands Islands."
        )

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {LOCAL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: LOCAL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-753",
                GEN_AI_RESPONSE_MODEL: LOCAL_MODEL,
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                GEN_AI_USAGE_INPUT_TOKENS: 52,
                GEN_AI_USAGE_OUTPUT_TOKENS: 11,
                SERVER_ADDRESS: "localhost",
                SERVER_PORT: 11434,
            },
        )
        self.assertEqual(span.events, ())

    def test_local_with_capture_content(self):
        # Redo the instrumentation dance to be affected by the environment variable
        OpenAIInstrumentor().uninstrument()
        with mock.patch.dict("os.environ", {"ELASTIC_OTEL_GENAI_CAPTURE_CONTENT": "true"}):
            OpenAIInstrumentor().instrument()
        client = openai.Client(base_url="http://localhost:11434/v1", api_key="unused", max_retries=1)

        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        chat_completion = client.chat.completions.create(model=LOCAL_MODEL, messages=messages)

        self.assertEqual(chat_completion.choices[0].message.content, "ocean A: Atlantic, B: Arctic Ocean")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {LOCAL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: LOCAL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-364",
                GEN_AI_RESPONSE_MODEL: LOCAL_MODEL,
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                GEN_AI_USAGE_INPUT_TOKENS: 52,
                GEN_AI_USAGE_OUTPUT_TOKENS: 11,
                SERVER_ADDRESS: "localhost",
                SERVER_PORT: 11434,
            },
        )
        self.assertEqual(len(span.events), 2)
        prompt_event, completion_event = span.events
        self.assertEqual(prompt_event.name, "gen_ai.content.prompt")
        self.assertEqual(dict(prompt_event.attributes), {"gen_ai.prompt": json.dumps(messages)})
        self.assertEqual(completion_event.name, "gen_ai.content.completion")
        self.assertEqual(
            dict(completion_event.attributes),
            {"gen_ai.completion": '[{"role": "assistant", "content": "ocean A: Atlantic, B: Arctic Ocean"}]'},
        )

    def test_stream(self):
        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        chat_completion = self.client.chat.completions.create(model=OPENAI_TOOL_MODEL, messages=messages, stream=True)

        chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
        self.assertEqual("".join(chunks), "South Atlantic Ocean.")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-A6Fj6kEv975Uw3vNCyA2njL0mP4Lg",
                GEN_AI_RESPONSE_MODEL: f"{OPENAI_TOOL_MODEL}-2024-07-18",
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )
        self.assertEqual(span.events, ())

        (operation_duration_metric,) = self.get_sorted_metrics()
        self.assertOperationDurationMetric(operation_duration_metric)

    def test_stream_with_include_usage_option(self):
        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        chat_completion = self.client.chat.completions.create(
            model=OPENAI_TOOL_MODEL, messages=messages, stream=True, stream_options={"include_usage": True}
        )

        chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
        self.assertEqual("".join(chunks), "South Atlantic Ocean.")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-A6Fj6QSKWN7eCCTYz5lKYkZMHngDq",
                GEN_AI_RESPONSE_MODEL: f"{OPENAI_TOOL_MODEL}-2024-07-18",
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                GEN_AI_USAGE_INPUT_TOKENS: 24,
                GEN_AI_USAGE_OUTPUT_TOKENS: 4,
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )
        self.assertEqual(span.events, ())

        operation_duration_metric, token_usage_metric = self.get_sorted_metrics()
        self.assertOperationDurationMetric(operation_duration_metric)
        self.assertTokenUsageMetric(token_usage_metric)

    def test_stream_with_tools_and_capture_content(self):
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

        chat_completion = self.client.chat.completions.create(
            model=OPENAI_TOOL_MODEL, messages=messages, tools=tools, stream=True
        )

        chunks = [chunk.choices[0].delta.content or "" for chunk in chat_completion if chunk.choices]
        self.assertEqual("".join(chunks), "")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-A86CT7lmQpCMrARhR2GkBP5JBYLfn",
                GEN_AI_RESPONSE_MODEL: f"{OPENAI_TOOL_MODEL}-2024-07-18",
                GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )

        self.assertEqual(len(span.events), 2)
        prompt_event, completion_event = span.events
        self.assertEqual(prompt_event.name, "gen_ai.content.prompt")
        self.assertEqual(dict(prompt_event.attributes), {"gen_ai.prompt": json.dumps(messages)})
        self.assertEqual(completion_event.name, "gen_ai.content.completion")
        self.assertEqual(
            dict(completion_event.attributes),
            {"gen_ai.completion": '[{"role": "assistant", "content": {"order_id": "order_12345"}}]'},
        )

    def test_local_stream(self):
        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        client = openai.Client(base_url="http://localhost:11434/v1", api_key="unused", max_retries=1)
        chat_completion = client.chat.completions.create(model=LOCAL_MODEL, messages=messages, stream=True)

        chunks = [chunk.choices[0].delta.content for chunk in chat_completion if chunk.choices]
        self.assertEqual("".join(chunks), "Oceania")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {LOCAL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: LOCAL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-735",
                GEN_AI_RESPONSE_MODEL: LOCAL_MODEL,
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                SERVER_ADDRESS: "localhost",
                SERVER_PORT: 11434,
            },
        )
        self.assertEqual(span.events, ())

    def test_local_stream_with_capture_content(self):
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

        client = openai.Client(base_url="http://localhost:11434/v1", api_key="unused", max_retries=1)
        chat_completion = client.chat.completions.create(model=LOCAL_MODEL, messages=messages, stream=True)

        chunks = [chunk.choices[0].delta.content for chunk in chat_completion if chunk.choices]
        self.assertEqual("".join(chunks), "Pacific Ocean.")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {LOCAL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: LOCAL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-829",
                GEN_AI_RESPONSE_MODEL: LOCAL_MODEL,
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                SERVER_ADDRESS: "localhost",
                SERVER_PORT: 11434,
            },
        )

        self.assertEqual(len(span.events), 2)
        prompt_event, completion_event = span.events
        self.assertEqual(prompt_event.name, "gen_ai.content.prompt")
        self.assertEqual(dict(prompt_event.attributes), {"gen_ai.prompt": json.dumps(messages)})
        self.assertEqual(completion_event.name, "gen_ai.content.completion")
        self.assertEqual(
            dict(completion_event.attributes),
            {"gen_ai.completion": ('[{"role": "assistant", "content": "Pacific Ocean."}]')},
        )

    def test_local_stream_error_handling(self):
        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        client = openai.Client(base_url="http://localhost:11434/v1", api_key="unused", max_retries=1)
        with mock.patch.object(openai._streaming.Stream, "__next__", side_effect=ValueError):
            chat_completion = client.chat.completions.create(model=LOCAL_MODEL, messages=messages, stream=True)
            with self.assertRaises(ValueError):
                [chunk.choices[0].delta.content for chunk in chat_completion if chunk.choices]

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {LOCAL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

        self.assertEqual(
            dict(span.attributes),
            {
                ERROR_TYPE: "ValueError",
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: LOCAL_MODEL,
                GEN_AI_SYSTEM: "openai",
                SERVER_ADDRESS: "localhost",
                SERVER_PORT: 11434,
            },
        )
        self.assertEqual(span.events, ())


class TestAsyncChatCompletions(OpenaiMixin, TestBase, IsolatedAsyncioTestCase):
    @classmethod
    def setup_client(cls):
        # Control the arguments
        return openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
            organization=os.getenv("OPENAI_ORG_ID", OPENAI_ORG_ID),
            project=os.getenv("OPENAI_PROJECT_ID", OPENAI_PROJECT_ID),
            max_retries=1,
        )

    async def test_async_basic(self):
        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        chat_completion = await self.client.chat.completions.create(model=OPENAI_TOOL_MODEL, messages=messages)

        self.assertEqual(chat_completion.choices[0].message.content, "South Atlantic Ocean.")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-A9CT0G8qhgAE0LVHYoClD0IG4eyKa",
                GEN_AI_RESPONSE_MODEL: OPENAI_TOOL_MODEL + "-2024-07-18",  # Note it is more specific than request!
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                GEN_AI_USAGE_INPUT_TOKENS: 24,
                GEN_AI_USAGE_OUTPUT_TOKENS: 4,
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )
        self.assertEqual(span.events, ())

        operation_duration_metric, token_usage_metric = self.get_sorted_metrics()
        self.assertOperationDurationMetric(operation_duration_metric)
        self.assertTokenUsageMetric(token_usage_metric)

    async def test_async_basic_with_capture_content(self):
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

        chat_completion = await self.client.chat.completions.create(model=OPENAI_TOOL_MODEL, messages=messages)

        self.assertEqual(chat_completion.choices[0].message.content, "South Atlantic Ocean.")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-A9CT2ePKnjnz40F1K7G6YhKMoNLsD",
                GEN_AI_RESPONSE_MODEL: OPENAI_TOOL_MODEL + "-2024-07-18",  # Note it is more specific than request!
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                GEN_AI_USAGE_INPUT_TOKENS: 24,
                GEN_AI_USAGE_OUTPUT_TOKENS: 4,
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )
        self.assertEqual(len(span.events), 2)
        prompt_event, completion_event = span.events
        self.assertEqual(prompt_event.name, "gen_ai.content.prompt")
        self.assertEqual(dict(prompt_event.attributes), {"gen_ai.prompt": json.dumps(messages)})
        self.assertEqual(completion_event.name, "gen_ai.content.completion")
        self.assertEqual(
            dict(completion_event.attributes),
            {"gen_ai.completion": ('[{"role": "assistant", "content": "South Atlantic Ocean."}]')},
        )

    async def test_async_stream(self):
        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        chat_completion = await self.client.chat.completions.create(
            model=OPENAI_TOOL_MODEL, messages=messages, stream=True
        )

        chunks = [chunk.choices[0].delta.content or "" async for chunk in chat_completion if chunk.choices]
        self.assertEqual("".join(chunks), "South Atlantic Ocean.")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-A6dGYxfNeE9JSyv3LWSkGTOIXJvqy",
                GEN_AI_RESPONSE_MODEL: f"{OPENAI_TOOL_MODEL}-2024-07-18",
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )
        self.assertEqual(span.events, ())

        (operation_duration_metric,) = self.get_sorted_metrics()
        self.assertOperationDurationMetric(operation_duration_metric)

    async def test_async_stream_with_capture_content(self):
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

        chat_completion = await self.client.chat.completions.create(
            model=OPENAI_TOOL_MODEL, messages=messages, stream=True
        )

        chunks = [chunk.choices[0].delta.content or "" async for chunk in chat_completion if chunk.choices]
        self.assertEqual("".join(chunks), "South Atlantic Ocean.")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-A6dGarmkMfJOnz2DWzymvkYpWQt00",
                GEN_AI_RESPONSE_MODEL: f"{OPENAI_TOOL_MODEL}-2024-07-18",
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )
        self.assertEqual(len(span.events), 2)
        prompt_event, completion_event = span.events
        self.assertEqual(prompt_event.name, "gen_ai.content.prompt")
        self.assertEqual(dict(prompt_event.attributes), {"gen_ai.prompt": json.dumps(messages)})
        self.assertEqual(completion_event.name, "gen_ai.content.completion")
        self.assertEqual(
            dict(completion_event.attributes),
            {"gen_ai.completion": ('[{"role": "assistant", "content": "South Atlantic Ocean."}]')},
        )

    async def test_async_tools_with_capture_content(self):
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

        response = await self.client.chat.completions.create(model=OPENAI_TOOL_MODEL, messages=messages, tools=tools)
        tool_call = response.choices[0].message.tool_calls[0]
        self.assertEqual(tool_call.function.name, "get_delivery_date")
        self.assertEqual(json.loads(tool_call.function.arguments), {"order_id": "order_12345"})

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-A9CT8USSOYj6tJBMsrClMJdppNCf3",
                GEN_AI_RESPONSE_MODEL: OPENAI_TOOL_MODEL + "-2024-07-18",  # Note it is more specific than request!
                GEN_AI_RESPONSE_FINISH_REASONS: ("tool_calls",),
                GEN_AI_USAGE_INPUT_TOKENS: 140,
                GEN_AI_USAGE_OUTPUT_TOKENS: 19,
                SERVER_ADDRESS: "api.openai.com",
                SERVER_PORT: 443,
            },
        )

        self.assertEqual(len(span.events), 2)
        prompt_event, completion_event = span.events
        self.assertEqual(prompt_event.name, "gen_ai.content.prompt")
        self.assertEqual(dict(prompt_event.attributes), {"gen_ai.prompt": json.dumps(messages)})
        self.assertEqual(completion_event.name, "gen_ai.content.completion")
        self.assertEqual(
            dict(completion_event.attributes),
            {"gen_ai.completion": '[{"role": "assistant", "content": {"order_id": "order_12345"}}]'},
        )

    async def test_async_local(self):
        client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="unused", max_retries=1)

        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        chat_completion = await client.chat.completions.create(model=LOCAL_MODEL, messages=messages)

        self.assertEqual(chat_completion.choices[0].message.content, "The South Pole is the answer.")

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {LOCAL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: LOCAL_MODEL,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_ID: "chatcmpl-533",
                GEN_AI_RESPONSE_MODEL: LOCAL_MODEL,
                GEN_AI_RESPONSE_FINISH_REASONS: ("stop",),
                GEN_AI_USAGE_INPUT_TOKENS: 52,
                GEN_AI_USAGE_OUTPUT_TOKENS: 8,
                SERVER_ADDRESS: "localhost",
                SERVER_PORT: 11434,
            },
        )
        self.assertEqual(span.events, ())

    async def test_async_local_stream_error_handling(self):
        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="unused", max_retries=1)
        with mock.patch.object(openai._streaming.AsyncStream, "__anext__", side_effect=ValueError):
            chat_completion = await client.chat.completions.create(model=LOCAL_MODEL, messages=messages, stream=True)
            with self.assertRaises(ValueError):
                [chunk.choices[0].delta.content async for chunk in chat_completion if chunk.choices]

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {LOCAL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

        self.assertEqual(
            dict(span.attributes),
            {
                ERROR_TYPE: "ValueError",
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: LOCAL_MODEL,
                GEN_AI_SYSTEM: "openai",
                SERVER_ADDRESS: "localhost",
                SERVER_PORT: 11434,
            },
        )
        self.assertEqual(span.events, ())

        (operation_duration_metric,) = self.get_sorted_metrics()
        self.assertErrorOperationDurationMetric(
            operation_duration_metric,
            {"gen_ai.request.model": LOCAL_MODEL, "error.type": "ValueError", "server.port": 11434},
            data_point=0.0032003200612962246,
        )

    async def test_async_local_connection_error(self):
        client = openai.AsyncOpenAI(base_url="http://localhost:9999/v5", api_key="unused", max_retries=1)
        messages = [
            {
                "role": "user",
                "content": "Answer in up to 3 words: Which ocean contains the falkland islands?",
            }
        ]

        with self.assertRaises(Exception):
            await client.chat.completions.create(model=OPENAI_TOOL_MODEL, messages=messages)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"chat {OPENAI_TOOL_MODEL}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: "chat",
                GEN_AI_REQUEST_MODEL: OPENAI_TOOL_MODEL,
                GEN_AI_SYSTEM: "openai",
                ERROR_TYPE: "APIConnectionError",
                SERVER_ADDRESS: "localhost",
                SERVER_PORT: 9999,
            },
        )
        self.assertEqual(span.events, ())

        (operation_duration_metric,) = self.get_sorted_metrics()
        self.assertErrorOperationDurationMetric(
            operation_duration_metric,
            {"error.type": "APIConnectionError"},
            data_point=0.0072969673201441765,
        )
