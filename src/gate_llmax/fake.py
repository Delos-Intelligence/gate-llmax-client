"""``FakeClient`` — an ``LLMClient`` that returns canned replies with no network.

For tests, demos, and UI plumbing: the full builder API works (``request(...).call(model)`` /
``.call_stream(model)`` / ``.cast_json()`` / ``.with_tools()`` …) — only the wire calls are
stubbed, so usage callbacks, budgets, rate limits and JSON parsing all run as normal.

This is a generic test double, distinct from any app's SSE/UI-protocol encoder.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from .client import LLMClient
from .models.response import LLMResponse, RawUsage, StreamChunk
from .types import OutputStatus

if TYPE_CHECKING:
    from .models.request import LLMRequest

Reply = str | Callable[["LLMRequest"], str]


class FakeClient(LLMClient):
    """A drop-in ``LLMClient`` that returns a canned reply without any HTTP.

    ``reply`` is a fixed string or a ``(LLMRequest) -> str`` callable. ``stream_word_by_word``
    controls whether ``call_stream`` emits one chunk per word (default) or the whole reply at once.
    """

    def __init__(
        self,
        reply: Reply = "",
        *,
        stream_word_by_word: bool = True,
        api_key: str = "fake",
        base_url: str = "http://fake",
        **kwargs: Any,
    ) -> None:
        """Build a fake client. Extra kwargs (usage_callback, budget, rate_limit, …) pass to ``LLMClient``."""
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._reply = reply
        self._stream_word_by_word = stream_word_by_word

    def _text_for(self, request: LLMRequest) -> str:
        reply = self._reply
        return reply if isinstance(reply, str) else reply(request)

    @staticmethod
    def _model_of(request: LLMRequest) -> str:
        return str(getattr(request.target, "model", "fake"))

    async def _send(self, request: LLMRequest, *, priority: int = 0) -> LLMResponse:
        model = self._model_of(request)
        return LLMResponse(
            raw_text=self._text_for(request),
            model=model,
            status=OutputStatus.SUCCESS,
            usage=RawUsage(model=model, operation=request.operation),
        )

    async def _stream(self, request: LLMRequest, *, priority: int = 0) -> AsyncIterator[StreamChunk]:
        text = self._text_for(request)
        if self._stream_word_by_word:
            words = text.split(" ")
            for i, word in enumerate(words):
                yield StreamChunk(text=word if i == len(words) - 1 else word + " ")
        elif text:
            yield StreamChunk(text=text)
        yield StreamChunk(is_done=True, input_tokens=0, output_tokens=0)
