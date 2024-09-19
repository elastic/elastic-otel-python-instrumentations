# Elastic Distribution of OpenTelemetry Python instrumentations

> [!WARNING]
> The Elastic Distribution of OpenTelemetry Python is not yet recommended for production use. Functionality may be changed or removed in future releases. Alpha releases are not subject to the support SLA of official GA features.
>
> We welcome your feedback! You can reach us by [opening a GitHub issue](https://github.com/elastic/elastic-otel-python-instrumentations/issues) or starting a discussion thread on the [Elastic Discuss forum](https://discuss.elastic.co/tags/c/observability/apm/58/python).

The Elastic Distribution of OpenTelemetry Python (EDOT Python) is a customized version of [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python).
EDOT Python makes it easier to get started using OpenTelemetry in your Python applications through strictly OpenTelemetry native means, while also providing a smooth and rich out of the box experience with [Elastic Observability](https://www.elastic.co/observability). It's an explicit goal of this distribution to introduce **no new concepts** in addition to those defined by the wider OpenTelemetry community.

This repository contains instrumentation libraries following the OpenTelemetry Semantic conventions.

## Available instrumentations

Each available instrumentation sits inside the `instrumentation` directory.

Currently we have:
- `elastic-opentelemetry-instrumentation-openai` for tracing the `openai` library

Please refer to each instrumentation `README.md` for details.

## License

This software is licensed under the Apache License, version 2 ("Apache-2.0").
