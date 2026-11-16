"""Read-only Cannery Hub bridge. No source downloads or build-time execution."""

import hashlib
import ipaddress
import json
import os
import re
import stat
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from time import monotonic
from types import ModuleType
from urllib.parse import urlsplit

import httpx
from huggingface_hub import constants, set_client_factory
from huggingface_hub.utils._http import default_client_factory

_PRIVATE_ENDPOINT = None

_DEADLINE: ContextVar[float | None] = ContextVar("cannery_deadline", default=None)


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or any((parsed.username, parsed.password, parsed.query, parsed.fragment))
    ):
        raise ValueError("Invalid private Hub endpoint")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname == "localhost"
    if parsed.scheme != "https" and not loopback:
        raise ValueError("Private Hub credentials require HTTPS or loopback")
    return f"{parsed.scheme}://{parsed.netloc.lower()}"


def _token(endpoint: str) -> str | bool:
    origin = _origin(endpoint)
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    directory = Path.home() / ".config/cannery/tokens"
    try:
        info = directory.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Cannery token directory must be private and user-owned")
    path = directory / (hashlib.sha256(origin.encode()).hexdigest() + ".json")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return False
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Cannery token must be a private user-owned regular file")
        with os.fdopen(descriptor, encoding="utf-8", closefd=False) as source:
            value = json.loads(source.read(4096))
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise ValueError("Invalid Cannery origin-bound token")
    token = value.get("token")
    if (
        value.get("origin") != origin
        or not isinstance(token, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token)
    ):
        raise ValueError("Invalid Cannery origin-bound token")
    return token


class _Stream(httpx.SyncByteStream):
    def __init__(self, stream, deadline, limit):
        self.stream, self.deadline, self.limit = stream, deadline, limit

    def __iter__(self):
        count = 0
        for chunk in self.stream:
            count += len(chunk)
            if monotonic() >= self.deadline or count > self.limit:
                raise ValueError("Private Hub transfer deadline or size exceeded")
            yield chunk

    def close(self):
        self.stream.close()


class _Transport(httpx.BaseTransport):
    def __init__(self, endpoint):
        self.origin = _origin(endpoint)
        self.transport = httpx.HTTPTransport(retries=0)

    def handle_request(self, request):
        if _origin(str(request.url.copy_with(query=None))) != self.origin:
            raise ValueError("Private Hub requests cannot leave the configured origin")
        deadline = _DEADLINE.get() or monotonic() + 60
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ValueError("Private Hub transfer deadline exceeded")
        request.extensions["timeout"] = dict.fromkeys(["connect", "read", "write", "pool"], min(15, remaining))
        response = self.transport.handle_request(request)
        limit = 128 * 1024 if "/api/kernels/" in request.url.path else 64 * 1024 * 1024
        response.stream = _Stream(response.stream, deadline, limit)
        return response

    def close(self):
        self.transport.close()


def configure_api(api):
    """Explicit private endpoints never inherit an unrelated cached Hub token."""
    global _PRIVATE_ENDPOINT
    endpoint = os.environ.get("HF_ENDPOINT")
    if not endpoint or _origin(endpoint) == "https://huggingface.co":
        if _PRIVATE_ENDPOINT is not None:
            set_client_factory(default_client_factory)
            _PRIVATE_ENDPOINT = None
        return api
    api.endpoint = endpoint.rstrip("/")
    api.token = _token(endpoint)
    if endpoint != _PRIVATE_ENDPOINT:
        set_client_factory(
            lambda: httpx.Client(
                transport=_Transport(endpoint),
                trust_env=False,
                follow_redirects=False,
            )
        )
        _PRIVATE_ENDPOINT = endpoint
    return api


def _bytes(path, limit):
    with open(path, "rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("Cannery artifact exceeds size limit")
    return value


@contextmanager
def _budget():
    handle = _DEADLINE.set(monotonic() + 60)
    try:
        yield
    finally:
        _DEADLINE.reset(handle)


def maybe_load(api, repo_id, revision, backend, trust_remote_code):
    """Return a Cannery module or None for ordinary Hub repositories."""
    endpoint = os.environ.get("HF_ENDPOINT")
    if (
        not endpoint
        or _origin(endpoint) == "https://huggingface.co"
        or revision is None
        or backend not in {None, "cann"}
    ):
        return None
    if constants.HF_HUB_OFFLINE:
        if revision.startswith("release-"):
            raise ValueError("Private Cannery get_kernel requires online publication verification")
        return None
    with _budget():
        info = api.repo_info(repo_id, repo_type="kernel", revision=revision)
        descriptor = getattr(info, "cannery", None)
        if descriptor is None:
            return None
        expected = {"schema", "repo_id", "release_id", "manifest_sha256", "package_sha256"}
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != expected
            or descriptor["schema"] != 1
            or descriptor["repo_id"] != repo_id
        ):
            raise ValueError("Unsupported Cannery Hub descriptor")
        for field in ["manifest_sha256", "package_sha256"]:
            if not isinstance(descriptor[field], str) or not re.fullmatch("[a-f0-9]{64}", descriptor[field]):
                raise ValueError("Invalid Cannery artifact pin")
        raw = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        commit = hashlib.sha1(b"cannery-hub-v1\0" + raw, usedforsecurity=False).hexdigest()
        if not isinstance(descriptor["release_id"], str):
            raise ValueError("Invalid Cannery release identity")
        if info.sha != commit or revision not in {commit, "release-" + descriptor["release_id"]}:
            raise ValueError("Cannery Hub revision mismatch")
        if not (trust_remote_code is True or isinstance(trust_remote_code, list) and repo_id in trust_remote_code):
            raise ValueError("Cannery native loading requires explicit trust_remote_code")
        from cannery_kernels import load_cannery_package

        payloads = {}
        for filename, field, limit in [
            ("release.json", "manifest_sha256", 128 * 1024),
            ("kernel.tar", "package_sha256", 64 * 1024 * 1024),
        ]:
            root = Path(os.environ.get("KERNELS_CACHE") or constants.HF_HUB_CACHE)
            cache = root / "cannery" / hashlib.sha256(_origin(api.endpoint).encode()).hexdigest()
            path = api.hf_hub_download(
                repo_id, filename=filename, repo_type="kernel", revision=commit, cache_dir=cache
            )
            content = _bytes(path, limit)
            if hashlib.sha256(content).hexdigest() != descriptor[field]:
                raise ValueError("Cannery artifact checksum mismatch")
            payloads[filename] = content
        manifest = json.loads(payloads["release.json"])
        if manifest["kernel"] != repo_id or manifest["package"]["sha256"] != descriptor["package_sha256"]:
            raise ValueError("Cannery manifest identity mismatch")
        loaded = load_cannery_package(payloads["kernel.tar"], descriptor["package_sha256"], trust_remote_code=True)
        module = ModuleType("cannery_kernel")
        module.forward = loaded.forward
        module._cannery_kernel = loaded
        return module
