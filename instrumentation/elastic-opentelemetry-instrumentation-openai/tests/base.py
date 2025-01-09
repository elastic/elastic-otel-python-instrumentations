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
from dataclasses import dataclass

import openai
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.metrics import Histogram
from vcr.unittest import VCRMixin

# Use the same model for tools as for chat completion
OPENAI_API_KEY = "test_openai_api_key"
OPENAI_ORG_ID = "test_openai_org_key"
OPENAI_PROJECT_ID = "test_openai_project_id"

LOCAL_MODEL = "qwen2.5:0.5b"


@dataclass
class OpenAIEnvironment:
    # TODO: add system
    operation_name: str = "chat"
    model: str = "gpt-4o-mini"
    response_model: str = "gpt-4o-mini-2024-07-18"
    server_address: str = "api.openai.com"
    server_port: int = 443


class OpenaiMixin(VCRMixin):
    def _get_vcr_kwargs(self, **kwargs):
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
    def setup_environment(cls):
        return OpenAIEnvironment()

    @classmethod
    def setUpClass(cls):
        cls.client = cls.setup_client()
        cls.openai_env = cls.setup_environment()

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
                        "gen_ai.operation.name": self.openai_env.operation_name,
                        "gen_ai.request.model": self.openai_env.model,
                        "gen_ai.response.model": self.openai_env.response_model,
                        "gen_ai.system": "openai",
                        "server.address": self.openai_env.server_address,
                        "server.port": self.openai_env.server_port,
                    },
                ),
            ],
            est_value_delta=0.2,
        )

    def assertErrorOperationDurationMetric(self, metric: Histogram, attributes: dict, data_point: float = None):
        self.assertEqual(metric.name, "gen_ai.client.operation.duration")
        default_attributes = {
            "gen_ai.operation.name": self.openai_env.operation_name,
            "gen_ai.request.model": self.openai_env.model,
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

    def assertTokenUsageInputMetric(self, metric: Histogram, input_data_point=4):
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
                        "gen_ai.operation.name": self.openai_env.operation_name,
                        "gen_ai.request.model": self.openai_env.model,
                        "gen_ai.response.model": self.openai_env.response_model,
                        "gen_ai.system": "openai",
                        "server.address": self.openai_env.server_address,
                        "server.port": self.openai_env.server_port,
                        "gen_ai.token.type": "input",
                    },
                ),
            ],
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
                        "gen_ai.operation.name": self.openai_env.operation_name,
                        "gen_ai.request.model": self.openai_env.model,
                        "gen_ai.response.model": self.openai_env.response_model,
                        "gen_ai.system": "openai",
                        "server.address": self.openai_env.server_address,
                        "server.port": self.openai_env.server_port,
                        "gen_ai.token.type": "input",
                    },
                ),
                self.create_histogram_data_point(
                    count=1,
                    sum_data_point=output_data_point,
                    max_data_point=output_data_point,
                    min_data_point=output_data_point,
                    attributes={
                        "gen_ai.operation.name": self.openai_env.operation_name,
                        "gen_ai.request.model": self.openai_env.model,
                        "gen_ai.response.model": self.openai_env.response_model,
                        "gen_ai.system": "openai",
                        "server.address": self.openai_env.server_address,
                        "server.port": self.openai_env.server_port,
                        "gen_ai.token.type": "output",
                    },
                ),
            ],
        )
