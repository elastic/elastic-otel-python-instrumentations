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

from opentelemetry.instrumentation.openai import OpenAIInstrumentor


def test_capture_message_content_false_by_default(instrument):
    instrument.uninstrument()
    assert not instrument.capture_message_content


def test_can_override_capture_message_content_programmatically(instrument):
    instrument.uninstrument()
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(capture_message_content=True)
    assert instrumentor.capture_message_content
    instrumentor.uninstrument()
