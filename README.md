# gate-llmax

Python client SDK and shared types for the [Gate LLM Gateway](https://github.com/Delos-Intelligence/gate-llmax-client) — one async API over OpenAI, Azure, Bedrock, Gemini and ElevenLabs.

## Installation

```bash
pip install "gate-llmax @ git+https://github.com/Delos-Intelligence/gate-llmax-client"
```

## How to use

Create a client pointing at your Gate instance:

```python
from gate_llmax import LLMClient

client = LLMClient(api_key="your-key", base_url="https://your-gate-instance.com")
```

Build a request and call a model:

```python
response = await client.request(prompt="Tell me a joke.").call("gpt-4o")
print(response.raw_text)
```

`request()` also takes a `system_prompt`, a full `messages` history, `images`, and per-call
`specifics` (temperature, tools, …). To stream the reply as it is generated, use `call_stream`:

```python
async for chunk in client.request(prompt="Tell me a joke.").call_stream("gpt-4o"):
    print(chunk.text, end="", flush=True)
```

The client owns an HTTP connection pool. Use it as an async context manager, or call
`await client.close()` when you are done:

```python
async with LLMClient(api_key="your-key", base_url="https://your-gate-instance.com") as client:
    response = await client.request(prompt="Hello!").call("gpt-4o")
```

## Shared types

The same package ships the Pydantic models the Gate backend and its clients exchange, so a
backend imports its data contracts straight from here:

```python
from gate_llmax.models.request import LLMRequest
from gate_llmax.models.response import LLMResponse
from gate_llmax.types import OutputStatus
```

## Requirements

Python 3.12+
