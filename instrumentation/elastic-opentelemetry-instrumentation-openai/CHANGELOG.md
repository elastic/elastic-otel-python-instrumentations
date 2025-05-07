# Elastic OpenTelemetry Instrumentation OpenAI

## v1.1.1

- Fix tracing of streamed `with_raw_response` API calls (#82)
- Test on Python 3.13 (#78)

## v1.1.0

- Fix missing or double spans when completion stream is used with context manager (#80)
- Follow semantic conventions 1.31.0: use GEN_AI_OUTPUT_TYPE, GEN_AI_REQUEST_SEED and GEN_AI_REQUEST_CHOICE_COUNT attributes (#77, #76, #75)

## v1.0.0

- Fix instrumentation of with_raw_response (#73)
- Use proper GenAI suggested buckets for metrics, requires 1.31.0+ SDK (#72)

## v0.7.0

- Map non-string, non-dict schema as json_schema (#68, Adrian Cole)
- Don't set attributes if they are NotGiven (#67)
- Apply instrumentation to beta.chat.Completions.parse (#65, Anuraag (Rag) Agrawal)
- Fix README venv installation command and dotenv (#63, Anuraag (Rag) Agrawal)

## v0.6.1

- Handle message with developer role (#58)
- Decouple metrics from span recording, fixes running with `OTEL_SDK_DISABLED=true` (#56)
- Add examples and simplify test matrix (#46, Adrian Cole)

## v0.6.0

- Trace GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT, GEN_AI_OPENAI_REQUEST_SEED, GEN_AI_OPENAI_REQUEST_SERVICE_TIER and GEN_AI_OPENAI_RESPONSE_SERVICE_TIER (#41)
- make OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT work like upstream (#42)

## v0.5.0

- Sync embeddings calls tracing with semantic conventions 1.29.0 (#36)
- Drop span events support for events, ELASTIC_OTEL_GENAI_EVENTS environment variable is gone (#38)
- Relax opentelemetry-api dependency so it supports 1.29.0+ (#40)

## v0.4.0

- Add support for tracing embeddings calls (#20)
- Rewrite tests in pytest style (#21)
- Don't crash on calls without a model (#22)
- Implement log events support following 1.28.0 GenAI semantic conventions (#23)
- Add tests for asserting exported schema version (#24)
- Don't update non-recording spans (#25)
- Add integration tests and test with both latest and baseline openai client (#28)
- Normalize capture content env variable to upstream (#29, Adrian Cole)
- Format recorded responses as upstream (#30, Adrian Cole)
- Use 1.28.0 semantic conventions log events as default (#31, Adrian Cole)
- Bump required api and sdk to 1.28.2/0.49b2 (#33)

## v0.3.0

- Delay loading of the openai module to close race condition with httpx instrumentation (#16)

## v0.2.0

- Fix tracing of optional OpenAI client parameters (#12)
- Fix schema versions to be strings (#9)
- Bump openai client we are testing against to latest (#11)

## v0.1.0

Initial release with the following features:
- `sync` and `async` chat completions
- Streaming support
- Functions calling with tools
- Client side metrics
- Following 1.27.0 [Gen AI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
