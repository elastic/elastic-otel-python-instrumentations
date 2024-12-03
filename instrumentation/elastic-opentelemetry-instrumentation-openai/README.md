# OpenTelemetry Instrumentation for OpenAI

An OpenTelemetry instrumentation for the `openai` client library.

This instrumentation currently supports instrumenting the chat completions and the embeddings APIs.

We currently support the following features:
- `sync` and `async` chat completions
- Streaming support for chat completions
- Functions calling with tools for chat completions
- Client side metrics
- Embeddings API calls
- Following 1.29.0 Gen AI Semantic Conventions

## Installation

```
pip install elastic-opentelemetry-instrumentation-openai
```

## Usage

This instrumentation supports *zero-code* / *autoinstrumentation*:

```
opentelemetry-instrument python use_openai.py

# You can record more information about prompts as log events by enabling content capture.
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true opentelemetry-instrument python use_openai.py
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

None

### Elastic specific semantic conventions

None at the moment

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

To refresh a test, delete its cassette file in tests/cassettes and make sure you have environment
variables set for recordings, detailed later.

If writing a new test, start with the test logic with no assertions. If extending an existing unit test
rather than writing a new one, remove the corresponding recorded response from [cassettes](./tests/cassettes/)
instead.

Then, run `pytest` as normal. It will execute a request against the LLM and record it. Update the
test with correct assertions until it passes. Following executions of `pytest` will use the recorded
response without querying the LLM.

### OpenAI Environment Variables

* `OPENAI_API_KEY` - from https://platform.openai.com/settings/profile?tab=api-keys
  * It should look like `sk-...` 

### Azure OpenAI Environment Variables

Azure is different from OpenAI primarily in that a URL has an implicit model. This means it ignores
the model parameter set by the OpenAI SDK. The implication is that one endpoint cannot serve both
chat and embeddings at the same time. Hence, we need separate environment variables for chat and
embeddings. In either case, the `DEPLOYMENT_URL` is the "Endpoint Target URI" and the `API_KEY` is
the `Endpoint Key` for a corresponding deployment in https://oai.azure.com/resource/deployments

* `AZURE_CHAT_COMPLETIONS_DEPLOYMENT_URL`
  * It should look like https://endpoint.com/openai/deployments/my-deployment/chat/completions?api-version=2023-05-15
* `AZURE_CHAT_COMPLETIONS_API_KEY`
  * It should be in hex like `abc01...` and possibly the same as `AZURE_EMBEDDINGS_API_KEY`
* `AZURE_EMBEDDINGS_DEPLOYMENT_URL`
  * It should look like https://endpoint.com/openai/deployments/my-deployment/embeddings?api-version=2023-05-15
* `AZURE_EMBEDDINGS_API_KEY`
  * It should be in hex like `abc01...` and possibly the same as `AZURE_CHAT_COMPLETIONS_API_KEY`

## License

This software is licensed under the Apache License, version 2 ("Apache-2.0").
