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
    _get_attributes_from_response,
    _record_operation_duration_metric,
    _record_token_usage_metrics,
    _send_log_events_from_stream_choices,
)
from opentelemetry.metrics import Histogram
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.types import Attributes
from wrapt import ObjectProxy

EVENT_GEN_AI_CONTENT_COMPLETION = "gen_ai.content.completion"

logger = logging.getLogger(__name__)


class StreamWrapper(ObjectProxy):
    def __init__(
        self,
        stream,
        span: Span,
        span_attributes: Attributes,
        capture_message_content: bool,
        event_attributes: Attributes,
        event_logger: EventLogger,
        start_time: float,
        token_usage_metric: Histogram,
        operation_duration_metric: Histogram,
    ):
        # we need to wrap the original response even in case of raw_responses
        super().__init__(stream)

        self.span = span
        self.span_attributes = span_attributes
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
        self.ended = False

    def end(self, exc=None):
        if self.ended:
            return

        self.ended = True
        if exc is not None:
            self.span.set_status(StatusCode.ERROR, str(exc))
            self.span.set_attribute(ERROR_TYPE, exc.__class__.__qualname__)
            self.span.end()
            error_attributes = {**self.span_attributes, ERROR_TYPE: exc.__class__.__qualname__}
            _record_operation_duration_metric(self.operation_duration_metric, error_attributes, self.start_time)
            return

        response_attributes = _get_attributes_from_response(
            self.response_id, self.model, self.choices, self.usage, self.service_tier
        )
        if self.span.is_recording():
            for k, v in response_attributes.items():
                self.span.set_attribute(k, v)

        metrics_attributes = {**self.span_attributes, **response_attributes}
        _record_operation_duration_metric(self.operation_duration_metric, metrics_attributes, self.start_time)
        if self.usage:
            _record_token_usage_metrics(self.token_usage_metric, metrics_attributes, self.usage)

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
        stream = self.__wrapped__
        try:
            for chunk in stream:
                self.process_chunk(chunk)
                yield chunk
        except Exception as exc:
            self.end(exc)
            raise
        self.end()

    async def __aenter__(self):
        # No difference in behavior between sync and async context manager
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.__exit__(exc_type, exc_value, traceback)

    async def __aiter__(self):
        stream = self.__wrapped__
        try:
            async for chunk in stream:
                self.process_chunk(chunk)
                yield chunk
        except Exception as exc:
            self.end(exc)
            raise
        self.end()

    def parse(self):
        """
        Handles direct parse() call on the client in order to maintain instrumentation on the parsed iterator.
        """
        parsed_iterator = self.__wrapped__.parse()

        parsed_wrapper = StreamWrapper(
            stream=parsed_iterator,
            span=self.span,
            span_attributes=self.span_attributes,
            capture_message_content=self.capture_message_content,
            event_attributes=self.event_attributes,
            event_logger=self.event_logger,
            start_time=self.start_time,
            token_usage_metric=self.token_usage_metric,
            operation_duration_metric=self.operation_duration_metric,
        )

        # Handle original sync/async iterators accordingly
        if hasattr(parsed_iterator, "__aiter__"):
            return parsed_wrapper.__aiter__()

        return parsed_wrapper.__iter__()
