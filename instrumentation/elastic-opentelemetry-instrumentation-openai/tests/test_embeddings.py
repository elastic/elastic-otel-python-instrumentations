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
from unittest import IsolatedAsyncioTestCase

import openai
from opentelemetry.instrumentation.openai.helpers import GEN_AI_REQUEST_ENCODING_FORMAT
from opentelemetry.test.test_base import TestBase
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

from .base import OPENAI_API_KEY, OpenAIEnvironment, OpenaiMixin


class TestOpenAIEmbeddings(OpenaiMixin, TestBase):
    @classmethod
    def setup_client(cls):
        return openai.Client(
            api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
            max_retries=1,
        )

    @classmethod
    def setup_environment(cls):
        return OpenAIEnvironment(
            model="text-embedding-3-small",
            response_model="text-embedding-3-small",
            operation_name="embeddings",
        )

    def test_basic(self):
        text = "South Atlantic Ocean."

        spans = self.get_finished_spans()
        response = self.client.embeddings.create(model=self.openai_env.model, input=[text])

        self.assertTrue(len(response.data), 1)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"embeddings {self.openai_env.model}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: self.openai_env.operation_name,
                GEN_AI_REQUEST_MODEL: self.openai_env.model,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_MODEL: self.openai_env.model,
                GEN_AI_USAGE_INPUT_TOKENS: 4,
                SERVER_ADDRESS: self.openai_env.server_address,
                SERVER_PORT: self.openai_env.server_port,
            },
        )
        self.assertEqual(span.events, ())

        operation_duration_metric, token_usage_metric = self.get_sorted_metrics()
        self.assertOperationDurationMetric(operation_duration_metric)
        self.assertTokenUsageInputMetric(token_usage_metric)

    def test_all_the_client_options(self):
        text = "South Atlantic Ocean."
        response = self.client.embeddings.create(model=self.openai_env.model, input=[text], encoding_format="float")

        self.assertTrue(len(response.data), 1)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"embeddings {self.openai_env.model}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: self.openai_env.operation_name,
                GEN_AI_REQUEST_MODEL: self.openai_env.model,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_MODEL: self.openai_env.model,
                GEN_AI_REQUEST_ENCODING_FORMAT: "float",
                GEN_AI_USAGE_INPUT_TOKENS: 4,
                SERVER_ADDRESS: self.openai_env.server_address,
                SERVER_PORT: self.openai_env.server_port,
            },
        )
        self.assertEqual(span.events, ())

        operation_duration_metric, token_usage_metric = self.get_sorted_metrics()
        self.assertOperationDurationMetric(operation_duration_metric)
        self.assertTokenUsageInputMetric(token_usage_metric)

    def test_connection_error(self):
        client = openai.Client(base_url="http://localhost:9999/v5", api_key="unused", max_retries=1)
        text = "South Atlantic Ocean."

        with self.assertRaises(Exception):
            client.embeddings.create(model=self.openai_env.model, input=[text])

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"embeddings {self.openai_env.model}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: self.openai_env.operation_name,
                GEN_AI_REQUEST_MODEL: self.openai_env.model,
                GEN_AI_SYSTEM: "openai",
                ERROR_TYPE: "APIConnectionError",
                SERVER_ADDRESS: "localhost",
                SERVER_PORT: 9999,
            },
        )
        self.assertEqual(span.events, ())

        (operation_duration_metric,) = self.get_sorted_metrics()
        self.assertErrorOperationDurationMetric(operation_duration_metric, {"error.type": "APIConnectionError"})


class TestAsyncOpenAIEmbeddings(OpenaiMixin, TestBase, IsolatedAsyncioTestCase):
    @classmethod
    def setup_client(cls):
        return openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", OPENAI_API_KEY),
            max_retries=1,
        )

    @classmethod
    def setup_environment(cls):
        return OpenAIEnvironment(
            model="text-embedding-3-small",
            response_model="text-embedding-3-small",
            operation_name="embeddings",
        )

    async def test_basic(self):
        text = "South Atlantic Ocean."

        spans = self.get_finished_spans()
        response = await self.client.embeddings.create(model=self.openai_env.model, input=[text])

        self.assertTrue(len(response.data), 1)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"embeddings {self.openai_env.model}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: self.openai_env.operation_name,
                GEN_AI_REQUEST_MODEL: self.openai_env.model,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_MODEL: self.openai_env.model,
                GEN_AI_USAGE_INPUT_TOKENS: 4,
                SERVER_ADDRESS: self.openai_env.server_address,
                SERVER_PORT: self.openai_env.server_port,
            },
        )
        self.assertEqual(span.events, ())

        operation_duration_metric, token_usage_metric = self.get_sorted_metrics()
        self.assertOperationDurationMetric(operation_duration_metric)
        self.assertTokenUsageInputMetric(token_usage_metric)

    async def test_all_the_client_options(self):
        text = "South Atlantic Ocean."
        response = await self.client.embeddings.create(
            model=self.openai_env.model, input=[text], encoding_format="float"
        )

        self.assertTrue(len(response.data), 1)

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"embeddings {self.openai_env.model}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: self.openai_env.operation_name,
                GEN_AI_REQUEST_MODEL: self.openai_env.model,
                GEN_AI_SYSTEM: "openai",
                GEN_AI_RESPONSE_MODEL: self.openai_env.model,
                GEN_AI_REQUEST_ENCODING_FORMAT: "float",
                GEN_AI_USAGE_INPUT_TOKENS: 4,
                SERVER_ADDRESS: self.openai_env.server_address,
                SERVER_PORT: self.openai_env.server_port,
            },
        )
        self.assertEqual(span.events, ())

        operation_duration_metric, token_usage_metric = self.get_sorted_metrics()
        self.assertOperationDurationMetric(operation_duration_metric)
        self.assertTokenUsageInputMetric(token_usage_metric)

    async def test_connection_error(self):
        client = openai.Client(base_url="http://localhost:9999/v5", api_key="unused", max_retries=1)
        text = "South Atlantic Ocean."

        with self.assertRaises(Exception):
            await client.embeddings.create(model=self.openai_env.model, input=[text])

        spans = self.get_finished_spans()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.name, f"embeddings {self.openai_env.model}")
        self.assertEqual(span.kind, SpanKind.CLIENT)
        self.assertEqual(span.status.status_code, StatusCode.ERROR)

        self.assertEqual(
            dict(span.attributes),
            {
                GEN_AI_OPERATION_NAME: self.openai_env.operation_name,
                GEN_AI_REQUEST_MODEL: self.openai_env.model,
                GEN_AI_SYSTEM: "openai",
                ERROR_TYPE: "APIConnectionError",
                SERVER_ADDRESS: "localhost",
                SERVER_PORT: 9999,
            },
        )
        self.assertEqual(span.events, ())

        (operation_duration_metric,) = self.get_sorted_metrics()
        self.assertErrorOperationDurationMetric(operation_duration_metric, {"error.type": "APIConnectionError"})


class TestLocalEmbeddings(TestOpenAIEmbeddings):
    @classmethod
    def setup_client(cls):
        return openai.Client(
            base_url="http://localhost:11434/v1",
            api_key="unused",
            max_retries=1,
        )

    @classmethod
    def setup_environment(cls):
        return OpenAIEnvironment(
            model="all-minilm:33m",
            response_model="all-minilm:33m",
            operation_name="embeddings",
            server_address="localhost",
            server_port=11434,
        )


class TestAsyncLocalEmbeddings(TestAsyncOpenAIEmbeddings):
    @classmethod
    def setup_client(cls):
        return openai.AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="unused",
            max_retries=1,
        )

    @classmethod
    def setup_environment(cls):
        return OpenAIEnvironment(
            model="all-minilm:33m",
            response_model="all-minilm:33m",
            operation_name="embeddings",
            server_address="localhost",
            server_port=11434,
        )
