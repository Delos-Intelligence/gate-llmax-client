"""``Message.from_openai`` preserves multimodal content (text + image_url) as Gate blocks."""

from __future__ import annotations

from gate_llmax.models.messages import ImageMessage, Message, MessageRole, TextMessage


def test_from_openai_preserves_data_uri_image() -> None:
    msg = Message.from_openai(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD", "detail": "high"}},
            ],
        },
    )
    assert msg.role == MessageRole.USER
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], TextMessage)
    assert msg.content[0].text == "what is this?"
    assert isinstance(msg.content[1], ImageMessage)
    assert msg.content[1].b64 == "QUJD"
    assert msg.content[1].detail == "high"


def test_from_openai_preserves_remote_image_url() -> None:
    msg = Message.from_openai(
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]},
    )
    assert isinstance(msg.content[0], ImageMessage)
    assert msg.content[0].url == "https://x/y.png"


def test_from_openai_plain_string_still_works() -> None:
    msg = Message.from_openai({"role": "user", "content": "hello"})
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextMessage)
    assert msg.content[0].text == "hello"
