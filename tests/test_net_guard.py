import pytest

from anther_ml import net_guard


class _BrokenResponse:
    is_redirect = False
    is_permanent_redirect = False

    def __init__(self):
        self.closed = False

    def iter_content(self, _chunk_size):
        yield b"partial"
        raise RuntimeError("connection reset")

    def close(self):
        self.closed = True


def test_safe_get_closes_response_when_body_iteration_fails(monkeypatch):
    response = _BrokenResponse()
    monkeypatch.setattr(net_guard, "validate_url", lambda url: url)
    monkeypatch.setattr(net_guard.requests, "get", lambda *a, **k: response)

    with pytest.raises(RuntimeError, match="connection reset"):
        net_guard.safe_get("https://p.scdn.co/preview")
    assert response.closed
