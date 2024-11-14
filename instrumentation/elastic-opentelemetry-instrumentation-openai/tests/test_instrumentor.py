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

from unittest import mock

import opentelemetry.instrumentation.openai
from opentelemetry.instrumentation.openai import OpenAIInstrumentor


def test_capture_content_false_by_default(instrument):
    instrument.uninstrument()
    assert not instrument.capture_content


def test_can_override_capture_content_programmatically(instrument):
    instrument.uninstrument()
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(capture_content=True)
    assert instrumentor.capture_content
    instrumentor.uninstrument()


def test_get_tracer_is_called_with_a_string_schema(instrument):
    instrument.uninstrument()
    instrumentor = OpenAIInstrumentor()
    with mock.patch.object(opentelemetry.instrumentation.openai, "get_tracer") as get_tracer_mock:
        instrumentor.instrument()
    instrumentor.uninstrument()
    get_tracer_mock.assert_called_once_with(
        "opentelemetry.instrumentation.openai", mock.ANY, None, schema_url="https://opentelemetry.io/schemas/1.27.0"
    )


def test_get_meter_is_called_with_a_string_schema(instrument):
    instrument.uninstrument()
    instrumentor = OpenAIInstrumentor()
    with mock.patch.object(opentelemetry.instrumentation.openai, "get_meter") as get_meter_mock:
        instrumentor.instrument()
    instrumentor.uninstrument()
    get_meter_mock.assert_called_once_with(
        "opentelemetry.instrumentation.openai", mock.ANY, None, schema_url="https://opentelemetry.io/schemas/1.27.0"
    )
