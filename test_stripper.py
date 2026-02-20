from main import ThinkingStripper


def _feed_all(chunks):
    s = ThinkingStripper()
    out = "".join(s.feed(c) for c in chunks)
    return out + s.flush()


def test_no_thinking():
    assert _feed_all(["hello world"]) == "hello world"


def test_thinking_single_chunk():
    assert _feed_all(["<thinking>hidden</thinking>visible"]) == "visible"


def test_thinking_split_open_tag():
    # opening tag split across chunks
    assert _feed_all(["<think", "ing>hidden</thinking>visible"]) == "visible"


def test_thinking_split_close_tag():
    # closing tag split across chunks
    assert _feed_all(["<thinking>hid", "den</think", "ing>visible"]) == "visible"


def test_thinking_at_end():
    assert _feed_all(["text<thinking>hidden</thinking>"]) == "text"


def test_multiple_thinking_blocks():
    result = _feed_all(["a<thinking>x</thinking>b<thinking>y</thinking>c"])
    assert result == "abc"


def test_thinking_across_many_chunks():
    chunks = list("<thinking>this is hidden</thinking>shown")
    assert _feed_all(chunks) == "shown"


def test_no_close_tag():
    # unclosed thinking block — flush should discard
    s = ThinkingStripper()
    out = s.feed("<thinking>never closed")
    assert out == ""
    assert s.flush() == ""


def test_plain_text_chunked():
    chunks = ["hel", "lo ", "wor", "ld"]
    assert _feed_all(chunks) == "hello world"
