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

from collections.abc import Iterable, Mapping
from timeit import default_timer
from typing import TYPE_CHECKING, Optional

from opentelemetry._events import Event, EventLogger
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT,
    GEN_AI_OPENAI_REQUEST_SEED,
    GEN_AI_OPENAI_REQUEST_SERVICE_TIER,
    GEN_AI_OPENAI_RESPONSE_SERVICE_TIER,
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_TOKEN_TYPE,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS, SERVER_PORT

try:
    from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_REQUEST_ENCODING_FORMATS
except ImportError:
    # available since 1.29.0
    GEN_AI_REQUEST_ENCODING_FORMATS = "gen_ai.request.encoding_formats"

from opentelemetry.metrics import Histogram
from opentelemetry.trace import Span
from opentelemetry.util.types import Attributes

EVENT_GEN_AI_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
EVENT_GEN_AI_CHOICE = "gen_ai.choice"
EVENT_GEN_AI_USER_MESSAGE = "gen_ai.user.message"
EVENT_GEN_AI_SYSTEM_MESSAGE = "gen_ai.system.message"
EVENT_GEN_AI_TOOL_MESSAGE = "gen_ai.tool.message"

# As this is only used for a type annotation, only import from openai module
# when running type checker like pyright since we otherwise don't want to import
# it before the app.
if TYPE_CHECKING:
    from openai.types import CompletionUsage
else:
    CompletionUsage = None


def _set_span_attributes_from_response(
    span: Span,
    response_id: str,
    model: str,
    choices,
    usage: CompletionUsage,
    service_tier: Optional[str],
) -> None:
    span.set_attribute(GEN_AI_RESPONSE_ID, response_id)
    span.set_attribute(GEN_AI_RESPONSE_MODEL, model)
    # when streaming finish_reason is None for every chunk that is not the last
    finish_reasons = [choice.finish_reason for choice in choices if choice.finish_reason]
    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons or ["error"])
    # without `include_usage` in `stream_options` we won't get this
    if usage:
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens)
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens)
    # this is available only if requested
    if service_tier:
        span.set_attribute(GEN_AI_OPENAI_RESPONSE_SERVICE_TIER, service_tier)


def _set_embeddings_span_attributes_from_response(span: Span, model: str, usage: CompletionUsage) -> None:
    span.set_attribute(GEN_AI_RESPONSE_MODEL, model)
    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens)


def _attributes_from_client(client) -> Attributes:
    span_attributes = {}

    if base_url := getattr(client, "_base_url", None):
        if host := getattr(base_url, "host", None):
            span_attributes[SERVER_ADDRESS] = host
        if port := getattr(base_url, "port", None):
            span_attributes[SERVER_PORT] = port
        elif scheme := getattr(base_url, "scheme", None):
            if scheme == "http":
                span_attributes[SERVER_PORT] = 80
            elif scheme == "https":
                span_attributes[SERVER_PORT] = 443

    return span_attributes


def _get_span_attributes_from_wrapper(instance, kwargs) -> Attributes:
    span_attributes = {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_SYSTEM: "openai",
    }

    if (request_model := kwargs.get("model")) is not None:
        span_attributes[GEN_AI_REQUEST_MODEL] = request_model

    if client := getattr(instance, "_client", None):
        span_attributes.update(_attributes_from_client(client))

    if (frequency_penalty := kwargs.get("frequency_penalty")) is not None:
        span_attributes[GEN_AI_REQUEST_FREQUENCY_PENALTY] = frequency_penalty
    if (max_tokens := kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))) is not None:
        span_attributes[GEN_AI_REQUEST_MAX_TOKENS] = max_tokens
    if (presence_penalty := kwargs.get("presence_penalty")) is not None:
        span_attributes[GEN_AI_REQUEST_PRESENCE_PENALTY] = presence_penalty
    if (temperature := kwargs.get("temperature")) is not None:
        span_attributes[GEN_AI_REQUEST_TEMPERATURE] = temperature
    if (top_p := kwargs.get("top_p")) is not None:
        span_attributes[GEN_AI_REQUEST_TOP_P] = top_p
    if (stop_sequences := kwargs.get("stop")) is not None:
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        span_attributes[GEN_AI_REQUEST_STOP_SEQUENCES] = stop_sequences
    if (seed := kwargs.get("seed")) is not None:
        span_attributes[GEN_AI_OPENAI_REQUEST_SEED] = seed
    if (service_tier := kwargs.get("service_tier")) is not None:
        span_attributes[GEN_AI_OPENAI_REQUEST_SERVICE_TIER] = service_tier
    if (response_format := kwargs.get("response_format")) is not None:
        # response_format may be string or object with a string in the `type` key
        if isinstance(response_format, Mapping):
            if (response_format_type := response_format.get("type")) is not None:
                span_attributes[GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT] = response_format_type
        else:
            span_attributes[GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT] = response_format

    return span_attributes


def _span_name_from_span_attributes(attributes: Attributes) -> str:
    request_model = attributes.get(GEN_AI_REQUEST_MODEL)
    return (
        f"{attributes[GEN_AI_OPERATION_NAME]} {request_model}"
        if request_model
        else f"{attributes[GEN_AI_OPERATION_NAME]}"
    )


def _get_embeddings_span_attributes_from_wrapper(instance, kwargs) -> Attributes:
    span_attributes = {
        GEN_AI_OPERATION_NAME: "embeddings",
        GEN_AI_SYSTEM: "openai",
    }

    if (request_model := kwargs.get("model")) is not None:
        span_attributes[GEN_AI_REQUEST_MODEL] = request_model

    if client := getattr(instance, "_client", None):
        span_attributes.update(_attributes_from_client(client))

    if (encoding_format := kwargs.get("encoding_format")) is not None:
        span_attributes[GEN_AI_REQUEST_ENCODING_FORMATS] = [encoding_format]

    return span_attributes


def _get_event_attributes() -> Attributes:
    return {GEN_AI_SYSTEM: "openai"}


def _get_attributes_if_set(span: Span, names: Iterable) -> Attributes:
    """Returns a dict with any attribute found in the span attributes"""
    attributes = span.attributes
    return {name: attributes[name] for name in names if name in attributes}


def _record_token_usage_metrics(metric: Histogram, span: Span, usage: CompletionUsage):
    token_usage_metric_attrs = _get_attributes_if_set(
        span,
        (
            GEN_AI_OPERATION_NAME,
            GEN_AI_REQUEST_MODEL,
            GEN_AI_RESPONSE_MODEL,
            GEN_AI_SYSTEM,
            SERVER_ADDRESS,
            SERVER_PORT,
        ),
    )
    metric.record(usage.prompt_tokens, {**token_usage_metric_attrs, GEN_AI_TOKEN_TYPE: "input"})
    # embeddings responses only have input tokens
    if hasattr(usage, "completion_tokens"):
        metric.record(usage.completion_tokens, {**token_usage_metric_attrs, GEN_AI_TOKEN_TYPE: "output"})


def _record_operation_duration_metric(metric: Histogram, span: Span, start: float):
    operation_duration_metric_attrs = _get_attributes_if_set(
        span,
        (
            GEN_AI_OPERATION_NAME,
            GEN_AI_REQUEST_MODEL,
            GEN_AI_RESPONSE_MODEL,
            GEN_AI_SYSTEM,
            ERROR_TYPE,
            SERVER_ADDRESS,
            SERVER_PORT,
        ),
    )
    duration_s = default_timer() - start
    metric.record(duration_s, operation_duration_metric_attrs)


def _key_or_property(obj, name):
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _serialize_tool_calls_for_event(tool_calls):
    return [
        {
            "id": _key_or_property(tool_call, "id"),
            "type": _key_or_property(tool_call, "type"),
            "function": {
                "name": _key_or_property(_key_or_property(tool_call, "function"), "name"),
                "arguments": _key_or_property(_key_or_property(tool_call, "function"), "arguments"),
            },
        }
        for tool_call in tool_calls
    ]


def _send_log_events_from_messages(
    event_logger: EventLogger, messages, attributes: Attributes, capture_message_content: bool
):
    for message in messages:
        body = {}
        if capture_message_content:
            content = message.get("content")
            if content:
                body["content"] = content
        if message["role"] == "system":
            event = Event(name=EVENT_GEN_AI_SYSTEM_MESSAGE, body=body, attributes=attributes)
            event_logger.emit(event)
        elif message["role"] == "user":
            event = Event(name=EVENT_GEN_AI_USER_MESSAGE, body=body, attributes=attributes)
            event_logger.emit(event)
        elif message["role"] == "assistant":
            tool_calls = _serialize_tool_calls_for_event(message.get("tool_calls", []))
            if tool_calls:
                body["tool_calls"] = tool_calls
            event = Event(
                name=EVENT_GEN_AI_ASSISTANT_MESSAGE,
                body=body,
                attributes=attributes,
            )
            event_logger.emit(event)
        elif message["role"] == "tool":
            body["id"] = message["tool_call_id"]
            event = Event(
                name=EVENT_GEN_AI_TOOL_MESSAGE,
                body=body,
                attributes=attributes,
            )
            event_logger.emit(event)


def _send_log_events_from_choices(
    event_logger: EventLogger, choices, attributes: Attributes, capture_message_content: bool
):
    for choice in choices:
        tool_calls = _serialize_tool_calls_for_event(choice.message.tool_calls or [])
        body = {"finish_reason": choice.finish_reason, "index": choice.index, "message": {}}
        if tool_calls:
            body["message"]["tool_calls"] = tool_calls
        if capture_message_content and choice.message.content:
            body["message"]["content"] = choice.message.content

        event = Event(name=EVENT_GEN_AI_CHOICE, body=body, attributes=attributes)
        event_logger.emit(event)


def _send_log_events_from_stream_choices(
    event_logger: EventLogger, choices, span: Span, attributes: Attributes, capture_message_content: bool
):
    body = {}
    message = {}
    message_content = ""
    tool_calls = {}
    for choice in choices:
        if choice.delta.content:
            message_content += choice.delta.content
        if choice.delta.tool_calls:
            for call in choice.delta.tool_calls:
                tool_calls.setdefault(call.index, {"function": {"arguments": ""}})
                tool_call = tool_calls[call.index]
                if call.function.arguments:
                    tool_call["function"]["arguments"] += call.function.arguments
                if call.function.name:
                    tool_call["function"]["name"] = call.function.name
                if call.id:
                    tool_call["id"] = call.id
                if call.type:
                    tool_call["type"] = call.type
        if choice.finish_reason:
            body["finish_reason"] = choice.finish_reason
        body["index"] = choice.index

    if capture_message_content and message_content:
        message["content"] = message_content
    if tool_calls:
        message["tool_calls"] = [call for _, call in sorted(tool_calls.items())]

    body = {
        "finish_reason": choice.finish_reason,
        "index": choice.index,
        "message": message,
    }
    # StreamWrapper is consumed after start_as_current_span exits, so capture the current span
    ctx = span.get_span_context()
    event = Event(
        name=EVENT_GEN_AI_CHOICE,
        body=body,
        attributes=attributes,
        trace_id=ctx.trace_id,
        span_id=ctx.span_id,
        trace_flags=ctx.trace_flags,
    )
    event_logger.emit(event)
