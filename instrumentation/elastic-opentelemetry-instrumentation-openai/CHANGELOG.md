# Elastic OpenTelemetry Instrumentation OpenAI

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
