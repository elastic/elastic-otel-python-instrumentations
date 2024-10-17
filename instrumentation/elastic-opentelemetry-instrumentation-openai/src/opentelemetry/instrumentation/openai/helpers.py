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
from collections.abc import Iterable
from timeit import default_timer
from typing import TYPE_CHECKING

from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS, SERVER_PORT
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_TOKEN_TYPE,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.metrics import Histogram
from opentelemetry.trace import Span

if TYPE_CHECKING:
    from openai.types import CompletionUsage
else:
    CompletionUsage = None


def _set_span_attributes_from_response(
    span: Span, response_id: str, model: str, choices, usage: CompletionUsage
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


def _decode_function_arguments(arguments: str):
    try:
        return json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return None


def _message_from_choice(choice):
    """Format a choice into a message of the same shape of the prompt"""
    if tool_calls := getattr(choice.message, "tool_calls", None):
        tool_call = tool_calls[0]
        return {"role": choice.message.role, "content": _decode_function_arguments(tool_call.function.arguments)}
    else:
        return {"role": choice.message.role, "content": choice.message.content}


def _message_from_stream_choices(choices):
    """Format an iterable of choices into a message of the same shape of the prompt"""
    message = {"role": None, "content": ""}
    tools_arguments = ""
    for choice in choices:
        if choice.delta.role:
            message["role"] = choice.delta.role
        if choice.delta.content:
            message["content"] += choice.delta.content
        if choice.delta.tool_calls:
            for call in choice.delta.tool_calls:
                if call.function.arguments:
                    tools_arguments += call.function.arguments

    if tools_arguments:
        if decoded_arguments := _decode_function_arguments(tools_arguments):
            message["content"] = decoded_arguments

    return message


def _get_span_attributes_from_wrapper(instance, kwargs):
    span_attributes = {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: kwargs["model"],
        GEN_AI_SYSTEM: "openai",
    }

    if client := getattr(instance, "_client", None):
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

    return span_attributes


def _get_attributes_if_set(span: Span, names: Iterable) -> dict:
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
