import pytest
from unittest.mock import patch


MOCK_RESP = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {"text": "Paris is the capital of France."},
                {"citationsContent": {"citations": [
                    {"location": {"web": {"url": "https://example.com/france"}}}
                ]}}
            ]
        }
    },
    "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
}


@pytest.fixture
def mock_bedrock():
    with patch("main.bedrock") as m:
        yield m


@pytest.fixture
def mock_resp():
    return MOCK_RESP
