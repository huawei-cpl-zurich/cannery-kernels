import hashlib
import json

import httpx
import pytest
from huggingface_hub import HfApi
from kernels import _cannery as bridge


def test_origin_bound_credentials(monkeypatch, tmp_path):
    endpoint = "http://localhost:8095"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    directory = tmp_path / ".config/cannery/tokens"
    directory.mkdir(parents=True, mode=0o700)
    file = directory / (hashlib.sha256(endpoint.encode()).hexdigest() + ".json")
    file.write_text(json.dumps({"origin": endpoint, "token": "x" * 40}))
    file.chmod(0o600)
    assert bridge._token(endpoint) == "x" * 40
    assert bridge._token("http://localhost:8096") is False
    monkeypatch.setenv("HF_TOKEN", "explicit")
    assert bridge._token(endpoint) == "explicit"
    monkeypatch.delenv("HF_TOKEN")
    file.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        bridge._token(endpoint)
    file.unlink()
    file.symlink_to(tmp_path / "absent")
    with pytest.raises(OSError):
        bridge._token(endpoint)
    with pytest.raises(ValueError, match="HTTPS"):
        bridge._token("http://remote.invalid")


def test_stream_size_and_origin_guards():
    with pytest.raises(ValueError, match="size"):
        list(bridge._Stream(httpx.ByteStream(b"abc"), 1))
    with bridge._Transport("http://localhost:8095") as transport:
        with pytest.raises(ValueError, match="origin"):
            transport.handle_request(
                httpx.Request("GET", "http://localhost:8096/private")
            )


@pytest.mark.parametrize(
    "status,headers",
    [(429, {"Retry-After": "3600"}), (200, {"Content-Encoding": "gzip"})],
)
def test_encoding_and_retry_after_cannot_bypass_limits(status, headers, monkeypatch):
    import time

    monkeypatch.setattr(
        time, "sleep", lambda _: pytest.fail("Unexpected upstream retry sleep")
    )
    with bridge._Transport("http://localhost:8095") as transport:
        transport.transport.close()
        transport.transport = httpx.MockTransport(
            lambda request: httpx.Response(
                status, headers=headers, stream=httpx.ByteStream(b"data")
            )
        )
        with httpx.Client(transport=transport) as client:
            with pytest.raises(ValueError, match="temporarily unavailable|Encoded"):
                client.get(
                    "http://localhost:8095/kernels/team/hello/resolve/revision/kernel.tar"
                )


def test_configuration_does_not_change_global_client(monkeypatch):
    from huggingface_hub.utils._http import get_session

    monkeypatch.setenv("HF_ENDPOINT", "http://localhost:8095")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    before = get_session()
    bridge.configure_api(HfApi())
    assert get_session() is before


def test_parent_bounds_worker_and_keeps_token_off_argv(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://localhost:8095")
    monkeypatch.setenv("HF_TOKEN", "test-token")

    def stalled(command, **kwargs):
        assert kwargs["timeout"] == 60
        assert "test-token" not in repr(command)
        raise bridge.subprocess.TimeoutExpired(command, 60)

    monkeypatch.setattr(bridge.subprocess, "run", stalled)
    with pytest.raises(ValueError, match="60-second deadline"):
        bridge.maybe_load(
            bridge.configure_api(HfApi()), "team/hello", "release-test", "cann", True
        )
