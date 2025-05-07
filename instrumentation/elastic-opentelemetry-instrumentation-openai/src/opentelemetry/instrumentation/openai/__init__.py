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
import os
from timeit import default_timer
from typing import Collection

from opentelemetry._events import get_event_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.openai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.instrumentation.openai.helpers import (
    _get_attributes_from_response,
    _get_attributes_from_wrapper,
    _get_embeddings_attributes_from_response,
    _get_embeddings_attributes_from_wrapper,
    _get_event_attributes,
    _is_raw_response,
    _record_operation_duration_metric,
    _record_token_usage_metrics,
    _send_log_events_from_choices,
    _send_log_events_from_messages,
    _span_name_from_attributes,
)
from opentelemetry.instrumentation.openai.metrics import (
    _GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
    _GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
)
from opentelemetry.instrumentation.openai.package import _instruments
from opentelemetry.instrumentation.openai.version import __version__
from opentelemetry.instrumentation.openai.wrappers import StreamWrapper
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.semconv._incubating.metrics.gen_ai_metrics import (
    GEN_AI_CLIENT_OPERATION_DURATION,
    GEN_AI_CLIENT_TOKEN_USAGE,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import StatusCode
from wrapt import register_post_import_hook, wrap_function_wrapper

EVENT_GEN_AI_CONTENT_PROMPT = "gen_ai.content.prompt"
EVENT_GEN_AI_CONTENT_COMPLETION = "gen_ai.content.completion"

logger = logging.getLogger(__name__)


class OpenAIInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Instruments OpenAI Completions and AsyncCompletions

        Args:
            **kwargs: Optional arguments
                ``tracer_provider``: a TracerProvider, defaults to global
                ``meter_provider``: a MeterProvider, defaults to global
                ``event_logger_provider``: a EventLoggerProvider, defaults to global
                ``capture_message_content``: to enable content capturing, defaults to False
        """
        capture_message_content = "true" if kwargs.get("capture_message_content") else "false"
        self.capture_message_content = (
            os.environ.get(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, capture_message_content).lower()
            == "true"
        )

        tracer_provider = kwargs.get("tracer_provider")
        self.tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_31_0.value,
        )
        meter_provider = kwargs.get("meter_provider")
        self.meter = get_meter(
            __name__,
            __version__,
            meter_provider,
            schema_url=Schemas.V1_31_0.value,
        )
        event_logger_provider = kwargs.get("event_logger_provider")
        self.event_logger = get_event_logger(__name__, event_logger_provider)

        self.token_usage_metric = self.meter.create_histogram(
            name=GEN_AI_CLIENT_TOKEN_USAGE,
            description="Measures number of input and output tokens used",
            unit="{token}",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_TOKEN_USAGE_BUCKETS,
        )

        self.operation_duration_metric = self.meter.create_histogram(
            name=GEN_AI_CLIENT_OPERATION_DURATION,
            description="GenAI operation duration",
            unit="s",
            explicit_bucket_boundaries_advisory=_GEN_AI_CLIENT_OPERATION_DURATION_BUCKETS,
        )

        register_post_import_hook(self._patch, "openai")

    def _patch(self, module):
        version = tuple([int(x) for x in getattr(getattr(module, "version"), "VERSION").split(".")])
        self.beta_chat_available = version >= (1, 40, 0)
        wrap_function_wrapper(
            "openai.resources.chat.completions",
            "Completions.create",
            self._chat_completion_wrapper,
        )
        wrap_function_wrapper(
            "openai.resources.chat.completions",
            "AsyncCompletions.create",
            self._async_chat_completion_wrapper,
        )
        if self.beta_chat_available:
            wrap_function_wrapper(
                "openai.resources.beta.chat.completions",
                "Completions.parse",
                self._chat_completion_wrapper,
            )
            wrap_function_wrapper(
                "openai.resources.beta.chat.completions",
                "AsyncCompletions.parse",
                self._async_chat_completion_wrapper,
            )
        wrap_function_wrapper(
            "openai.resources.embeddings",
            "Embeddings.create",
            self._embeddings_wrapper,
        )
        wrap_function_wrapper(
            "openai.resources.embeddings",
            "AsyncEmbeddings.create",
            self._async_embeddings_wrapper,
        )

    def _uninstrument(self, **kwargs):
        # unwrap only supports uninstrumenting real module references so we
        # import here.
        import openai

        unwrap(openai.resources.chat.completions.Completions, "create")
        unwrap(openai.resources.chat.completions.AsyncCompletions, "create")
        if self.beta_chat_available:
            unwrap(openai.resources.beta.chat.completions.Completions, "parse")
            unwrap(openai.resources.beta.chat.completions.AsyncCompletions, "parse")
        unwrap(openai.resources.embeddings.Embeddings, "create")
        unwrap(openai.resources.embeddings.AsyncEmbeddings, "create")

    def _chat_completion_wrapper(self, wrapped, instance, args, kwargs):
        logger.debug(f"{wrapped} kwargs: {kwargs}")

        span_attributes = _get_attributes_from_wrapper(instance, kwargs)
        event_attributes = _get_event_attributes()

        span_name = _span_name_from_attributes(span_attributes)
        with self.tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            # this is important to avoid having the span closed before ending the stream
            end_on_exit=False,
        ) as span:
            messages = kwargs.get("messages", [])
            _send_log_events_from_messages(
                self.event_logger,
                messages=messages,
                attributes=event_attributes,
                capture_message_content=self.capture_message_content,
            )

            start_time = default_timer()
            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.set_attribute(ERROR_TYPE, exc.__class__.__qualname__)
                span.end()
                error_attributes = {**span_attributes, ERROR_TYPE: exc.__class__.__qualname__}
                _record_operation_duration_metric(self.operation_duration_metric, error_attributes, start_time)
                raise

            if kwargs.get("stream"):
                return StreamWrapper(
                    stream=result,
                    span=span,
                    span_attributes=span_attributes,
                    capture_message_content=self.capture_message_content,
                    event_attributes=event_attributes,
                    event_logger=self.event_logger,
                    start_time=start_time,
                    token_usage_metric=self.token_usage_metric,
                    operation_duration_metric=self.operation_duration_metric,
                )

            logger.debug(f"openai.resources.chat.completions.Completions.create result: {result}")

            # if the caller is using with_raw_response we need to parse the output to get the response class we expect
            is_raw_response = _is_raw_response(result)
            if is_raw_response:
                result = result.parse()
            response_attributes = _get_attributes_from_response(
                result.id, result.model, result.choices, result.usage, getattr(result, "service_tier", None)
            )
            if span.is_recording():
                for k, v in response_attributes.items():
                    span.set_attribute(k, v)

            metrics_attributes = {**span_attributes, **response_attributes}
            _record_token_usage_metrics(self.token_usage_metric, metrics_attributes, result.usage)
            _record_operation_duration_metric(self.operation_duration_metric, metrics_attributes, start_time)

            _send_log_events_from_choices(
                self.event_logger,
                choices=result.choices,
                attributes=event_attributes,
                capture_message_content=self.capture_message_content,
            )

            span.end()

            return result

    async def _async_chat_completion_wrapper(self, wrapped, instance, args, kwargs):
        logger.debug(f"openai.resources.chat.completions.AsyncCompletions.create kwargs: {kwargs}")

        span_attributes = _get_attributes_from_wrapper(instance, kwargs)
        event_attributes = _get_event_attributes()

        span_name = _span_name_from_attributes(span_attributes)
        with self.tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            # this is important to avoid having the span closed before ending the stream
            end_on_exit=False,
        ) as span:
            messages = kwargs.get("messages", [])
            _send_log_events_from_messages(
                self.event_logger,
                messages=messages,
                attributes=event_attributes,
                capture_message_content=self.capture_message_content,
            )

            start_time = default_timer()
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.set_attribute(ERROR_TYPE, exc.__class__.__qualname__)
                span.end()
                error_attributes = {ERROR_TYPE: exc.__class__.__qualname__}
                _record_operation_duration_metric(self.operation_duration_metric, error_attributes, start_time)
                raise

            if kwargs.get("stream"):
                return StreamWrapper(
                    stream=result,
                    span=span,
                    span_attributes=span_attributes,
                    capture_message_content=self.capture_message_content,
                    event_attributes=event_attributes,
                    event_logger=self.event_logger,
                    start_time=start_time,
                    token_usage_metric=self.token_usage_metric,
                    operation_duration_metric=self.operation_duration_metric,
                )

            logger.debug(f"openai.resources.chat.completions.AsyncCompletions.create result: {result}")

            # if the caller is using with_raw_response we need to parse the output to get the response class we expect
            is_raw_response = _is_raw_response(result)
            if is_raw_response:
                result = result.parse()
            response_attributes = _get_attributes_from_response(
                result.id, result.model, result.choices, result.usage, getattr(result, "service_tier", None)
            )
            if span.is_recording():
                for k, v in response_attributes.items():
                    span.set_attribute(k, v)

            metrics_attributes = {**span_attributes, **response_attributes}
            _record_token_usage_metrics(self.token_usage_metric, metrics_attributes, result.usage)
            _record_operation_duration_metric(self.operation_duration_metric, metrics_attributes, start_time)

            _send_log_events_from_choices(
                self.event_logger,
                choices=result.choices,
                attributes=event_attributes,
                capture_message_content=self.capture_message_content,
            )

            span.end()

            return result

    def _embeddings_wrapper(self, wrapped, instance, args, kwargs):
        span_attributes = _get_embeddings_attributes_from_wrapper(instance, kwargs)

        span_name = _span_name_from_attributes(span_attributes)
        with self.tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            # this is important to avoid having the span closed before ending the stream
            end_on_exit=False,
        ) as span:
            start_time = default_timer()
            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.set_attribute(ERROR_TYPE, exc.__class__.__qualname__)
                span.end()
                error_attributes = {**span_attributes, ERROR_TYPE: exc.__class__.__qualname__}
                _record_operation_duration_metric(self.operation_duration_metric, error_attributes, start_time)
                raise

            response_attributes = _get_embeddings_attributes_from_response(result.model, result.usage)
            if span.is_recording():
                for k, v in response_attributes.items():
                    span.set_attribute(k, v)

            metrics_attributes = {**span_attributes, **response_attributes}
            _record_token_usage_metrics(self.token_usage_metric, metrics_attributes, result.usage)
            _record_operation_duration_metric(self.operation_duration_metric, metrics_attributes, start_time)

            span.end()

            return result

    async def _async_embeddings_wrapper(self, wrapped, instance, args, kwargs):
        span_attributes = _get_embeddings_attributes_from_wrapper(instance, kwargs)

        span_name = _span_name_from_attributes(span_attributes)
        with self.tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            # this is important to avoid having the span closed before ending the stream
            end_on_exit=False,
        ) as span:
            start_time = default_timer()
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.set_attribute(ERROR_TYPE, exc.__class__.__qualname__)
                span.end()
                error_attributes = {**span_attributes, ERROR_TYPE: exc.__class__.__qualname__}
                _record_operation_duration_metric(self.operation_duration_metric, error_attributes, start_time)
                raise

            response_attributes = _get_embeddings_attributes_from_response(result.model, result.usage)
            if span.is_recording():
                for k, v in response_attributes.items():
                    span.set_attribute(k, v)

            metrics_attributes = {**span_attributes, **response_attributes}
            _record_token_usage_metrics(self.token_usage_metric, metrics_attributes, result.usage)
            _record_operation_duration_metric(self.operation_duration_metric, metrics_attributes, start_time)

            span.end()

            return result
