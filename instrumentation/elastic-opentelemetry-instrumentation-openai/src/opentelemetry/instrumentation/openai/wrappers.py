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

import logging

from opentelemetry._events import EventLogger
from opentelemetry.instrumentation.openai.helpers import (
    _record_operation_duration_metric,
    _record_token_usage_metrics,
    _send_log_events_from_stream_choices,
    _set_span_attributes_from_response,
)
from opentelemetry.metrics import Histogram
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.types import Attributes

EVENT_GEN_AI_CONTENT_COMPLETION = "gen_ai.content.completion"

logger = logging.getLogger(__name__)


class StreamWrapper:
    def __init__(
        self,
        stream,
        span: Span,
        capture_message_content: bool,
        event_attributes: Attributes,
        event_logger: EventLogger,
        start_time: float,
        token_usage_metric: Histogram,
        operation_duration_metric: Histogram,
    ):
        self.stream = stream
        self.span = span
        self.capture_message_content = capture_message_content
        self.event_attributes = event_attributes
        self.event_logger = event_logger
        self.token_usage_metric = token_usage_metric
        self.operation_duration_metric = operation_duration_metric
        self.start_time = start_time

        self.response_id = None
        self.model = None
        self.choices = []
        self.usage = None
        self.service_tier = None

    def end(self, exc=None):
        # StopIteration is not an error, it signals that we have consumed all the stream
        if exc is not None and not isinstance(exc, (StopIteration, StopAsyncIteration)):
            self.span.set_status(StatusCode.ERROR, str(exc))
            self.span.set_attribute(ERROR_TYPE, exc.__class__.__qualname__)
            self.span.end()
            _record_operation_duration_metric(self.operation_duration_metric, self.span, self.start_time)
            return

        if self.span.is_recording():
            _set_span_attributes_from_response(
                self.span, self.response_id, self.model, self.choices, self.usage, self.service_tier
            )

        _record_operation_duration_metric(self.operation_duration_metric, self.span, self.start_time)
        if self.usage:
            _record_token_usage_metrics(self.token_usage_metric, self.span, self.usage)

        _send_log_events_from_stream_choices(
            self.event_logger,
            choices=self.choices,
            span=self.span,
            attributes=self.event_attributes,
            capture_message_content=self.capture_message_content,
        )

        self.span.end()

    def process_chunk(self, chunk):
        self.response_id = chunk.id
        self.model = chunk.model
        # usage with streaming is available since 1.26.0
        if hasattr(chunk, "usage"):
            self.usage = chunk.usage
        # with `include_usage` in `stream_options` we will get a last chunk without choices
        if chunk.choices:
            self.choices += chunk.choices
        if hasattr(chunk, "service_tier"):
            self.service_tier = chunk.service_tier

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end(exc_value)

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.stream)
            self.process_chunk(chunk)
            return chunk
        except Exception as exc:
            self.end(exc)
            raise

    async def __anext__(self):
        try:
            chunk = await self.stream.__anext__()
            self.process_chunk(chunk)
            return chunk
        except Exception as exc:
            self.end(exc)
            raise
