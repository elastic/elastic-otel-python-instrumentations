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
import logging
import os
from timeit import default_timer
from typing import Collection

from wrapt import register_post_import_hook, wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.instrumentation.openai.environment_variables import (
    ELASTIC_OTEL_GENAI_CAPTURE_CONTENT,
)
from opentelemetry.instrumentation.openai.helpers import (
    _get_span_attributes_from_wrapper,
    _message_from_choice,
    _record_token_usage_metrics,
    _record_operation_duration_metric,
    _set_span_attributes_from_response,
)
from opentelemetry.instrumentation.openai.package import _instruments
from opentelemetry.instrumentation.openai.version import __version__
from opentelemetry.instrumentation.openai.wrappers import StreamWrapper
from opentelemetry.metrics import get_meter
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_COMPLETION,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROMPT,
    GEN_AI_REQUEST_MODEL,
)
from opentelemetry.semconv._incubating.metrics.gen_ai_metrics import (
    create_gen_ai_client_token_usage,
    create_gen_ai_client_operation_duration,
)

from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import StatusCode

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
                ``capture_content``: to enable content capturing, defaults to False
        """
        capture_content = "true" if kwargs.get("capture_content") else "false"
        self.capture_content = os.environ.get(ELASTIC_OTEL_GENAI_CAPTURE_CONTENT, capture_content).lower() == "true"
        tracer_provider = kwargs.get("tracer_provider")
        self.tracer = get_tracer(__name__, __version__, tracer_provider, schema_url=Schemas.V1_27_0.value)
        meter_provider = kwargs.get("meter_provider")
        self.meter = get_meter(__name__, __version__, meter_provider, schema_url=Schemas.V1_27_0.value)
        self.token_usage_metric = create_gen_ai_client_token_usage(self.meter)
        self.operation_duration_metric = create_gen_ai_client_operation_duration(self.meter)

        register_post_import_hook(self._patch, "openai")

    def _patch(self, _module):
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

    def _uninstrument(self, **kwargs):
        # unwrap only supports uninstrementing real module references so we
        # import here.
        import openai

        unwrap(openai.resources.chat.completions.Completions, "create")
        unwrap(openai.resources.chat.completions.AsyncCompletions, "create")

    def _chat_completion_wrapper(self, wrapped, instance, args, kwargs):
        logger.debug(f"openai.resources.chat.completions.Completions.create kwargs: {kwargs}")

        span_attributes = _get_span_attributes_from_wrapper(instance, kwargs)

        span_name = f"{span_attributes[GEN_AI_OPERATION_NAME]} {span_attributes[GEN_AI_REQUEST_MODEL]}"
        with self.tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            # this is important to avoid having the span closed before ending the stream
            end_on_exit=False,
        ) as span:
            if self.capture_content:
                messages = kwargs.get("messages", [])
                prompt = [message for message in messages]
                try:
                    span.add_event(EVENT_GEN_AI_CONTENT_PROMPT, attributes={GEN_AI_PROMPT: json.dumps(prompt)})
                except TypeError:
                    logger.error(f"Failed to serialize {EVENT_GEN_AI_CONTENT_PROMPT}")

            start_time = default_timer()
            try:
                result = wrapped(*args, **kwargs)
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.set_attribute(ERROR_TYPE, exc.__class__.__qualname__)
                span.end()
                _record_operation_duration_metric(self.operation_duration_metric, span, start_time)
                raise

            if kwargs.get("stream"):
                return StreamWrapper(
                    stream=result,
                    span=span,
                    capture_content=self.capture_content,
                    start_time=start_time,
                    token_usage_metric=self.token_usage_metric,
                    operation_duration_metric=self.operation_duration_metric,
                )

            logger.debug(f"openai.resources.chat.completions.Completions.create result: {result}")

            _set_span_attributes_from_response(span, result.id, result.model, result.choices, result.usage)

            _record_token_usage_metrics(self.token_usage_metric, span, result.usage)
            _record_operation_duration_metric(self.operation_duration_metric, span, start_time)

            if self.capture_content:
                # same format as the prompt
                completion = [_message_from_choice(choice) for choice in result.choices]
                try:
                    span.add_event(
                        EVENT_GEN_AI_CONTENT_COMPLETION, attributes={GEN_AI_COMPLETION: json.dumps(completion)}
                    )
                except TypeError:
                    logger.error(f"Failed to serialize {EVENT_GEN_AI_CONTENT_COMPLETION}")

            span.end()

            return result

    async def _async_chat_completion_wrapper(self, wrapped, instance, args, kwargs):
        logger.debug(f"openai.resources.chat.completions.AsyncCompletions.create kwargs: {kwargs}")

        span_attributes = _get_span_attributes_from_wrapper(instance, kwargs)

        span_name = f"{span_attributes[GEN_AI_OPERATION_NAME]} {span_attributes[GEN_AI_REQUEST_MODEL]}"
        with self.tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
            attributes=span_attributes,
            # this is important to avoid having the span closed before ending the stream
            end_on_exit=False,
        ) as span:
            if self.capture_content:
                messages = kwargs.get("messages", [])
                try:
                    span.add_event(EVENT_GEN_AI_CONTENT_PROMPT, attributes={GEN_AI_PROMPT: json.dumps(messages)})
                except TypeError:
                    logger.error(f"Failed to serialize {EVENT_GEN_AI_CONTENT_PROMPT}")

            start_time = default_timer()
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.set_attribute(ERROR_TYPE, exc.__class__.__qualname__)
                span.end()
                _record_operation_duration_metric(self.operation_duration_metric, span, start_time)
                raise

            if kwargs.get("stream"):
                return StreamWrapper(
                    stream=result,
                    span=span,
                    capture_content=self.capture_content,
                    start_time=start_time,
                    token_usage_metric=self.token_usage_metric,
                    operation_duration_metric=self.operation_duration_metric,
                )

            logger.debug(f"openai.resources.chat.completions.AsyncCompletions.create result: {result}")

            _set_span_attributes_from_response(span, result.id, result.model, result.choices, result.usage)

            _record_token_usage_metrics(self.token_usage_metric, span, result.usage)
            _record_operation_duration_metric(self.operation_duration_metric, span, start_time)

            if self.capture_content:
                # same format as the prompt
                completion = [_message_from_choice(choice) for choice in result.choices]
                try:
                    span.add_event(
                        EVENT_GEN_AI_CONTENT_COMPLETION, attributes={GEN_AI_COMPLETION: json.dumps(completion)}
                    )
                except TypeError:
                    logger.error(f"Failed to serialize {EVENT_GEN_AI_CONTENT_COMPLETION}")

            span.end()

            return result
