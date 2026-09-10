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

import openai
import pytest

TEST_CHAT_MODEL = "gpt-4o-mini"
TEST_CHAT_INPUT = "Answer in up to 3 words: Which ocean contains Bouvet Island?"
TEST_EMBEDDINGS_MODEL = "text-embedding-3-small"
TEST_EMBEDDINGS_INPUT = "South Atlantic Ocean."


@pytest.fixture
def vcr_cassette_name(request):
    return request.node.name.replace("_with_otel_sdk_disabled", "")


@pytest.mark.vcr()
def test_chat_with_otel_sdk_disabled(default_openai_env, trace_exporter, otel_sdk_disabled_instrument):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=TEST_CHAT_MODEL,
        messages=[{"role": "user", "content": TEST_CHAT_INPUT}],
    )

    assert response.choices[0].message.content == "Atlantic Ocean."
    assert not trace_exporter.get_finished_spans()


@pytest.mark.vcr()
def test_embeddings_with_otel_sdk_disabled(default_openai_env, trace_exporter, otel_sdk_disabled_instrument):
    client = openai.OpenAI()

    response = client.embeddings.create(model=TEST_EMBEDDINGS_MODEL, input=[TEST_EMBEDDINGS_INPUT])

    assert len(response.data) == 1
    assert not trace_exporter.get_finished_spans()
