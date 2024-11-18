# OpenTelemetry Instrumentation for OpenAI

An OpenTelemetry instrumentation for the `openai` client library.

This instrumentation currently only supports instrumenting the Chat completions APIs.

We currently support the following features:
- `sync` and `async` chat completions
- Streaming support
- Functions calling with tools
- Client side metrics
- Following 1.27.0 Gen AI Semantic Conventions

## Installation

```
pip install elastic-opentelemetry-instrumentation-openai
```

## Usage

This instrumentation supports *0-code* / *autoinstrumentation*:

```
opentelemetry-instrument python use_openai.py

# You can record more information about prompts as span events by enabling content capture.
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true opentelemetry-instrument python use_openai.py

# You can record more information about prompts as log events by enabling content capture.
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true ELASTIC_OTEL_GENAI_EVENTS=log opentelemetry-instrument python use_openai.py
```

Or manual instrumentation:

```python
import openai
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

OpenAIInstrumentor().instrument()

# assumes at least the OPENAI_API_KEY environment variable set
client = openai.Client()

messages = [
    {
        "role": "user",
        "content": "Answer in up to 3 words: Which ocean contains the canarian islands?",
    }
]

chat_completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
```

### Instrumentation specific environment variable configuration

- `ELASTIC_OTEL_GENAI_EVENTS` (default: `span`): when set to `log` exports GenAI events as
log events instead of span events.

### Elastic specific semantic conventions

- New `embeddings` value for `gen_ai.operation.name`
- New `gen_ai.request.encoding_format` attribute with openai specific values `[float, base64]`

## Development

We use [pytest](https://docs.pytest.org/en/stable/) to execute tests written with the standard
library [unittest](https://docs.python.org/3/library/unittest.html) framework.

Test dependencies need to be installed before running.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r dev-requirements.txt

pytest
```

To run integration tests doing real requests:

```
OPENAI_API_KEY=unused pytest --integration-tests
```

## Refreshing HTTP payloads

We use [VCR.py](https://vcrpy.readthedocs.io/en/latest/) to automatically record HTTP responses from
LLMs to reuse in tests without running the LLM. Refreshing HTTP payloads may be needed in these
cases

- Adding a new unit test
- Extending a unit test with functionality that requires an up-to-date HTTP response

Integration tests default to using ollama, to avoid cost and leaking sensitive information.
However, unit test recordings should use the authoritative OpenAI platform unless the test is
about a specific portability corner case.

To refresh a test, delete its cassette file in tests/cassettes and make sure you have the
following environment variables set for OpenAI recordings:

* `OPENAI_API_KEY` - from https://platform.openai.com/settings/profile?tab=api-keys
  * It should look like `sk-...` 

and the following for Azure recordings, deployment url and api key are available for each deployment
in https://oai.azure.com/resource/deployments:

* `AZURE_CHAT_COMPLETIONS_DEPLOYMENT_URL`
* `AZURE_CHAT_COMPLETIONS_API_KEY`
* `AZURE_EMBEDDINGS_DEPLOYMENT_URL`
* `AZURE_EMBEDDINGS_API_KEY`

If writing a new test, start with the test logic with no assertions. If extending an existing unit test
rather than writing a new one, remove the corresponding recorded response from [cassettes](./tests/cassettes/)
instead.

Then, run `pytest` as normal. It will execute a request against the LLM and record it. Update the
test with correct assertions until it passes. Following executions of `pytest` will use the recorded
response without querying the LLM.

## License

This software is licensed under the Apache License, version 2 ("Apache-2.0").
