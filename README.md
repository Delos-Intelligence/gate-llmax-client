# gate-llmax

Python client SDK **and** shared types for the [Gate LLM Gateway](https://github.com/Delos-Intelligence/gate-llmax-client) — a
unified API layer over multiple LLM providers (OpenAI, Azure, Bedrock, Gemini, ElevenLabs).

This package ships two import modules:

- **`gate`** — the ergonomic async client SDK (`GateClient`, request builders, streaming, token counting).
- **`gate_common`** — the shared Pydantic models and type aliases (`GateRequest`, `GateResponse`, `Message`,
  `ModelInfo`, `StreamChunk`, …). These are the data contracts the Gate backend and any client share, so backends
  depend on this package for types too.

## Install

```bash
pip install "gate-llmax @ git+https://github.com/Delos-Intelligence/gate-llmax-client"
```

or with uv:

```toml
[project]
dependencies = ["gate-llmax"]

[tool.uv.sources]
gate-llmax = { git = "https://github.com/Delos-Intelligence/gate-llmax-client", branch = "main" }
```

## Usage

```python
from gate import GateClient

async with GateClient(base_url="https://your-gate-instance.com", api_key="your-key") as client:
    response = await client.request(prompt="Hello!").call("gpt-4o")
    print(response.raw_text)
```

Streaming:

```python
async for chunk in client.request(prompt="...").call_stream("gpt-4o"):
    print(chunk.delta, end="", flush=True)
```

Importing shared types directly (e.g. from a backend):

```python
from gate_common.models.request import GateRequest
from gate_common.models.response import GateResponse
from gate_common.types import OutputStatus
```

## Development

```bash
uv sync --group dev
uv run pytest tests          # unit tests run offline; test_client.py needs a live Gate server (see tests/.env.example)
uv run ruff check src tests
```

## Requirements

Python 3.12+
