import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_basic_completion(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    resp = client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "Capital of France?"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Paris is the capital of France."
    assert data["citations"] == ["https://example.com/france"]
    assert data["usage"]["total_tokens"] == 30


def test_model_mapping_sonar(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "sonar",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert mock_bedrock.converse.call_args[1]["modelId"] == "us.amazon.nova-premier-v1:0"


def test_model_mapping_sonar_pro(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert mock_bedrock.converse.call_args[1]["modelId"] == "us.amazon.nova-premier-v1:0"


def test_system_message_split(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "hi"},
        ],
    })
    kwargs = mock_bedrock.converse.call_args[1]
    assert kwargs["system"] == [{"text": "Be concise."}]
    assert len(kwargs["messages"]) == 1
    assert kwargs["messages"][0]["role"] == "user"


def test_thinking_stripped(mock_bedrock):
    mock_bedrock.converse.return_value = {
        "output": {"message": {"content": [
            {"text": "<thinking>internal</thinking>The answer is 42."}
        ]}},
        "usage": {},
    }
    resp = client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "?"}],
    })
    assert resp.json()["choices"][0]["message"]["content"] == "The answer is 42."


def test_content_parts(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    resp = client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
    })
    assert resp.status_code == 200
    assert mock_bedrock.converse.call_args[1]["messages"][0]["content"][0]["text"] == "Hello"


def test_temperature_forwarded(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
    })
    cfg = mock_bedrock.converse.call_args[1]["inferenceConfig"]
    assert cfg["temperature"] == 0.7
    assert cfg["topP"] == 0.9


def test_throttling_returns_429(mock_bedrock):
    from botocore.exceptions import ClientError
    mock_bedrock.converse.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "Converse"
    )
    resp = client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 429
    assert "Rate exceeded" in resp.json()["error"]["message"]


def test_access_denied_returns_403(mock_bedrock):
    from botocore.exceptions import ClientError
    mock_bedrock.converse.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "Not authorized"}},
        "Converse"
    )
    resp = client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 403


def test_generic_error_returns_500(mock_bedrock):
    mock_bedrock.converse.side_effect = Exception("unexpected")
    resp = client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 500
    assert "unexpected" in resp.json()["error"]["message"]


def test_streaming(mock_bedrock):
    events = [
        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
        {"contentBlockDelta": {"delta": {"text": " world"}}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 10, "totalTokens": 15}}},
    ]
    mock_bedrock.converse_stream.return_value = {"stream": iter(events)}

    with client.stream("POST", "/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as resp:
        assert resp.status_code == 200
        chunks = []
        for line in resp.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))

    text = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c.get("choices") and c["choices"][0].get("delta")
    )
    assert text == "Hello world"


def test_streaming_client_error(mock_bedrock):
    from botocore.exceptions import ClientError
    mock_bedrock.converse_stream.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "ConverseStream"
    )
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as resp:
        assert resp.status_code == 200
        lines = [l for l in resp.iter_lines() if l.startswith("data: ") and l != "data: [DONE]"]
        error_chunk = json.loads(lines[-1][6:])
        assert "error" in error_chunk
        assert error_chunk["error"]["code"] == 429


def test_max_completion_tokens(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 512,
    })
    assert mock_bedrock.converse.call_args[1]["inferenceConfig"]["maxTokens"] == 512


def test_max_completion_tokens_overrides_max_tokens(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
        "max_completion_tokens": 512,
    })
    assert mock_bedrock.converse.call_args[1]["inferenceConfig"]["maxTokens"] == 512


def test_response_format_accepted(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    resp = client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {"type": "json_object"},
    })
    assert resp.status_code == 200


def test_get_model():
    resp = client.get("/v1/models/sonar-pro")
    assert resp.status_code == 200
    assert resp.json()["id"] == "sonar-pro"


def test_get_model_not_found():
    resp = client.get("/v1/models/nonexistent-model")
    assert resp.status_code == 200
    assert resp.json()["root"] == "us.amazon.nova-premier-v1:0"


def test_finish_reason_max_tokens(mock_bedrock):
    mock_bedrock.converse.return_value = {
        "output": {"message": {"content": [{"text": "truncated"}]}},
        "stopReason": "max_tokens",
        "usage": {},
    }
    resp = client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.json()["choices"][0]["finish_reason"] == "length"


def test_finish_reason_end_turn(mock_bedrock):
    mock_bedrock.converse.return_value = {
        "output": {"message": {"content": [{"text": "done"}]}},
        "stopReason": "end_turn",
        "usage": {},
    }
    resp = client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.json()["choices"][0]["finish_reason"] == "stop"


def test_streaming_finish_reason_max_tokens(mock_bedrock):
    events = [
        {"contentBlockDelta": {"delta": {"text": "partial"}}},
        {"messageStop": {"stopReason": "max_tokens"}},
        {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 10, "totalTokens": 15}}},
    ]
    mock_bedrock.converse_stream.return_value = {"stream": iter(events)}

    with client.stream("POST", "/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as resp:
        chunks = [
            json.loads(line[6:])
            for line in resp.iter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]

    final = next(c for c in chunks if c.get("choices", [{}])[0].get("finish_reason"))
    assert final["choices"][0]["finish_reason"] == "length"


def test_logprobs_accepted(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    resp = client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": True,
        "top_logprobs": 3,
    })
    assert resp.status_code == 200


def test_auth_no_key_required_by_default(mock_bedrock, mock_resp):
    """Without API_KEY set, any key (or none) is accepted."""
    mock_bedrock.converse.return_value = mock_resp
    resp = client.post("/v1/chat/completions",
        headers={"Authorization": "Bearer anything"},
        json={"model": "sonar-pro", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200


def test_auth_enforced_when_key_set(mock_bedrock, mock_resp):
    import main
    original = main._API_KEY
    try:
        main._API_KEY = "secret"
        resp = client.post("/v1/chat/completions",
            headers={"Authorization": "Bearer wrong"},
            json={"model": "sonar-pro", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401
    finally:
        main._API_KEY = original


def test_auth_passes_with_correct_key(mock_bedrock, mock_resp):
    import main
    original = main._API_KEY
    try:
        main._API_KEY = "secret"
        mock_bedrock.converse.return_value = mock_resp
        resp = client.post("/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={"model": "sonar-pro", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
    finally:
        main._API_KEY = original


def test_health_always_unauthenticated():
    assert client.get("/health").json() == {"status": "ok"}


def test_model_mapping_sonar_mini(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "sonar-mini",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert mock_bedrock.converse.call_args[1]["modelId"] == "us.amazon.nova-premier-v1:0"


def test_model_mapping_sonar_turbo(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "sonar-turbo",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert mock_bedrock.converse.call_args[1]["modelId"] == "us.amazon.nova-premier-v1:0"


def test_stop_sequences_forwarded(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stop": ["END", "STOP"],
    })
    cfg = mock_bedrock.converse.call_args[1]["inferenceConfig"]
    assert cfg["stopSequences"] == ["END", "STOP"]


def test_stop_string_normalized(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stop": "END",
    })
    cfg = mock_bedrock.converse.call_args[1]["inferenceConfig"]
    assert cfg["stopSequences"] == ["END"]


def test_list_models():
    data = client.get("/v1/models").json()
    ids = [m["id"] for m in data["data"]]
    for expected in ("sonar-pro", "sonar-pro-online", "sonar", "sonar-online",
                     "sonar-reasoning", "sonar-deep-research", "sonar-turbo", "sonar-mini"):
        assert expected in ids


def test_unknown_model_falls_back_to_premier(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert mock_bedrock.converse.call_args[1]["modelId"] == "us.amazon.nova-premier-v1:0"


def test_default_max_tokens(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert mock_bedrock.converse.call_args[1]["inferenceConfig"]["maxTokens"] == 4096


def test_chat_completions_alias(mock_bedrock, mock_resp):
    mock_bedrock.converse.return_value = mock_resp
    resp = client.post("/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["role"] == "assistant"


def test_rate_limit_returns_429():
    from unittest.mock import patch, AsyncMock
    with patch("main._try_acquire", new=AsyncMock(return_value=False)):
        resp = client.post("/v1/chat/completions", json={
            "model": "sonar-pro",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert resp.status_code == 429
    assert resp.headers.get("Retry-After") == "5"
    assert "rate_limit_error" in resp.json()["error"]["type"]


def test_rate_limit_semaphore_released_after_request(mock_bedrock, mock_resp):
    import main
    mock_bedrock.converse.return_value = mock_resp
    before = main._semaphore._value
    client.post("/v1/chat/completions", json={
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert main._semaphore._value == before
