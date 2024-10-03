# Elastic OpenTelemetry Instrumentation OpenAI

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
