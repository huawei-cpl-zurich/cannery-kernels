"""Real HF HTTP and get_kernel routing; native execution is explicitly mocked on CPU."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from huggingface_hub import HfApi

import cannery_kernels
from kernels import get_kernel
from kernels import _cannery as bridge


@pytest.fixture
def hub(monkeypatch, tmp_path):
    package = b"fixture native package"
    manifest = json.dumps(
        {
            "kernel": "team/hello",
            "package": {"sha256": hashlib.sha256(package).hexdigest()},
        }
    ).encode()
    descriptor = {
        "schema": 1,
        "repo_id": "team/hello",
        "release_id": str(uuid4()),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "package_sha256": hashlib.sha256(package).hexdigest(),
    }
    commit = hashlib.sha1(
        b"cannery-hub-v1\0"
        + json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    state = SimpleNamespace(
        descriptor=descriptor, commit=commit, status=200, corrupt=False, requests=[]
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            state.requests.append((self.command, self.path))
            if self.headers.get("Authorization") != "Bearer " + "x" * 40:
                self.send_error(401)
                return
            if state.status != 200:
                self.send_error(state.status)
                return
            if self.path.startswith("/api/kernels/"):
                value = json.dumps(
                    {
                        "id": "team/hello",
                        "sha": state.commit,
                        "cannery": state.descriptor,
                    }
                ).encode()
                etag = hashlib.sha256(value).hexdigest()
            else:
                name = self.path.rsplit("/", 1)[-1]
                value = manifest if name == "release.json" else package
                etag = descriptor[
                    "manifest_sha256" if name == "release.json" else "package_sha256"
                ]
                if state.corrupt and name == "kernel.tar":
                    value = b"corrupt"
            self.send_response(200)
            self.send_header("Content-Length", str(len(value)))
            self.send_header("X-Repo-Commit", state.commit)
            self.send_header("ETag", '"' + etag + '"')
            self.end_headers()
            if self.command == "GET":
                self.wfile.write(value)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("HF_ENDPOINT", endpoint)
    monkeypatch.setenv("HF_TOKEN", "x" * 40)
    monkeypatch.setenv("KERNELS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setattr(
        cannery_kernels,
        "load_cannery_package",
        lambda *a, **k: SimpleNamespace(forward=lambda x: x + 1),
    )
    try:
        yield state, endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        monkeypatch.delenv("HF_ENDPOINT")
        bridge.configure_api(HfApi())


def load(state, **kwargs):
    return get_kernel(
        "team/hello",
        revision="release-" + state.descriptor["release_id"],
        backend="cann",
        **kwargs,
    )


def test_get_kernel_cold_warm_and_withdrawal(hub):
    state, _ = hub
    first = load(state, trust_remote_code=True)
    assert first.forward(2) == 3 and first._cannery_kernel is not None
    assert load(state, trust_remote_code=["team/hello"]).forward(3) == 4
    state.status = 404
    with pytest.raises(Exception):
        load(state, trust_remote_code=True)


def test_trust_and_corruption_fail_before_native_loading(hub, monkeypatch):
    state, _ = hub
    calls = []
    monkeypatch.setattr(
        cannery_kernels, "load_cannery_package", lambda *a, **k: calls.append(a)
    )
    with pytest.raises(ValueError, match="explicit trust"):
        load(state, trust_remote_code=False)
    assert not any(path.endswith("kernel.tar") for method, path in state.requests)
    state.corrupt = True
    with pytest.raises(ValueError, match="checksum"):
        load(state, trust_remote_code=True)
    assert not calls


def test_backend_error_recovers_without_native_fallback(hub):
    state, _ = hub
    state.status = 503
    with pytest.raises(Exception):
        load(state, trust_remote_code=True)
    state.status = 200
    assert load(state, trust_remote_code=True).forward(1) == 2


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


def test_stream_deadline_size_and_origin_guards(monkeypatch):
    monkeypatch.setattr(bridge, "monotonic", lambda: 10)
    for deadline, limit in [(9, 100), (20, 1)]:
        with pytest.raises(ValueError, match="deadline or size"):
            list(bridge._Stream(httpx.ByteStream(b"abc"), deadline, limit))
    with bridge._Transport("http://localhost:8095") as transport:
        with pytest.raises(ValueError, match="origin"):
            transport.handle_request(
                httpx.Request("GET", "http://localhost:8096/private")
            )


def test_offline_and_incompatible_runtime_are_explicit(hub, monkeypatch):
    state, _ = hub
    monkeypatch.setattr(bridge.constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(ValueError, match="online"):
        load(state, trust_remote_code=True)
    monkeypatch.setattr(bridge.constants, "HF_HUB_OFFLINE", False)

    def incompatible(*args, **kwargs):
        raise cannery_kernels.CanneryPackageError(
            "Package runtime fingerprint mismatch"
        )

    monkeypatch.setattr(cannery_kernels, "load_cannery_package", incompatible)
    with pytest.raises(cannery_kernels.CanneryPackageError, match="fingerprint"):
        load(state, trust_remote_code=True)


def test_ordinary_repository_keeps_upstream_loading(hub, monkeypatch):
    import importlib

    state, _ = hub
    state.descriptor = None
    module = importlib.import_module("kernels.load")
    monkeypatch.setattr(module, "get_kernel_with_resolver", lambda **kwargs: "upstream")
    assert get_kernel("team/hello", revision="a" * 40, backend="cann") == "upstream"


def test_public_and_ordinary_offline_keep_upstream(hub, monkeypatch):
    import importlib

    module = importlib.import_module("kernels.load")
    monkeypatch.setattr(module, "get_kernel_with_resolver", lambda **kwargs: "cached")
    monkeypatch.setattr(bridge.constants, "HF_HUB_OFFLINE", True)
    assert get_kernel("team/ordinary", revision="a" * 40) == "cached"
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    assert get_kernel("team/ordinary", revision="a" * 40) == "cached"


def test_local_override_does_not_probe_failed_endpoint(hub, monkeypatch, tmp_path):
    import importlib

    state, _ = hub
    state.status = 503
    monkeypatch.setenv("LOCAL_KERNELS", "team/hello=" + str(tmp_path))
    module = importlib.import_module("kernels.load")
    monkeypatch.setattr(module, "get_kernel_with_resolver", lambda **kwargs: "local")
    assert load(state, trust_remote_code=True) == "local"
    assert state.requests == []


def test_same_origin_query_is_allowed(hub):
    _, endpoint = hub
    with httpx.Client(transport=bridge._Transport(endpoint)) as client:
        assert (
            client.get(endpoint + "/api/kernels/team/hello?recursive=false").status_code
            == 401
        )
