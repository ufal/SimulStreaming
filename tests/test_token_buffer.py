"""Test token buffer functionality."""
import pytest
from token_buffer import TokenBuffer


def test_token_buffer_creation():
    """Test TokenBuffer can be created."""
    buffer = TokenBuffer(text="hello", prefix_token_ids=[1, 2, 3])

    assert buffer is not None
    assert buffer.text == "hello"
    assert buffer.prefix_token_ids == [1, 2, 3]


def test_token_buffer_empty():
    """Test creating empty buffer."""
    buffer = TokenBuffer.empty()

    assert buffer.is_empty()
    assert buffer.text == ""


def test_token_buffer_from_text():
    """Test creating buffer from text."""
    buffer = TokenBuffer.from_text("test text")

    assert buffer.text == "test text"
    assert not buffer.is_empty()


def test_token_buffer_as_text():
    """Test getting text from buffer."""
    buffer = TokenBuffer(text="hello world")

    result = buffer.as_text()
    assert result == "hello world"


def test_token_buffer_prefix_tokens():
    """Test buffer with prefix tokens."""
    prefix = [100, 101, 102]
    buffer = TokenBuffer(text="test", prefix_token_ids=prefix)

    assert buffer.prefix_token_ids == prefix
    assert buffer.text == "test"
