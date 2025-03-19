# OpenTelemetry Instrumentation for OpenAI

An OpenTelemetry instrumentation for the `openai` client library.

This instrumentation currently supports instrumenting the chat completions (create and parse (beta) APIs) and the embeddings API.

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

Set up a virtual environment with this package, the dependencies it requires
and `dotenv` (a portable way to load environment variables).
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r dev-requirements.txt
pip install python-dotenv[cli]
```

Create a `.env` file containing the OpenAI API key:

```
echo "OPENAI_API_KEY=sk-..." > .env
```

Run the script with telemetry setup to use the instrumentation.

```
dotenv run -- opentelemetry-instrument python examples/chat.py
```

You can record more information about prompts as log events by enabling content capture.
```
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true dotenv run -- \
opentelemetry-instrument python examples/chat.py
```

### Using a local model

[Ollama](https://ollama.com/) may be used to run examples without a cloud account. After you have set it up
need to install the models in order to run the examples:

```
# for chat
ollama pull qwen2.5:0.5b
# for embeddings
ollama pull all-minilm:33m
```

Finally run the examples using [ollama.env](ollama.env) variables to point to Ollama instead of OpenAI:

```
dotenv -f ollama.env run -- opentelemetry-instrument python examples/chat.py
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

The `AzureOpenAI` client extends `OpenAI` with parameters specific to the Azure OpenAI Service.

* `AZURE_OPENAI_ENDPOINT` - "Azure OpenAI Endpoint" in https://oai.azure.com/resource/overview
  * It should look like `https://<your-resource-name>.openai.azure.com/`
* `AZURE_OPENAI_API_KEY` - "API key 1 (or 2)" in https://oai.azure.com/resource/overview
  * It should look be a hex string like `abc01...`
* `OPENAI_API_VERSION` = "Inference version" from https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation
  * It should look like `2024-10-01-preview`
* `TEST_CHAT_MODEL` = "Name" from https://oai.azure.com/resource/deployments that deployed a model
  that supports tool calling, such as "gpt-4o-mini".
* `TEST_EMBEDDINGS_MODEL` = "Name" from https://oai.azure.com/resource/deployments that deployed a
  model that supports embeddings, such as "text-embedding-3-small".

Note: The model parameter of a chat completion or embeddings request is substituted for an identical
deployment name. As deployment names are arbitrary they may have no correlation with a real model
like `gpt-4o`

## License

This software is licensed under the Apache License, version 2 ("Apache-2.0").
