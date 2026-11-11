"""Integration checks for server construction, tracing and OAuth storage."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastmcp", reason="feast[mcp-server] extra not installed")

from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from feast.mcp import server  # noqa: E402
from feast.mcp.config import (  # noqa: E402
    AuthConfig,
    Config,
    ServerConfig,
    SessionStorageConfig,
)
from feast.mcp.observability.logger import request_id_var  # noqa: E402


def test_oidc_receives_configured_session_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = object()
    build_store = MagicMock(return_value=store)
    create_auth = MagicMock()
    monkeypatch.setattr(server, "_build_session_store", build_store)
    monkeypatch.setattr(server, "create_oidc_auth", create_auth)
    cfg = Config(
        auth=AuthConfig(
            mode="oidc",
            discovery_url="https://issuer.example/.well-known/openid-configuration",
            client_id="feast-mcp",
        ),
        session_storage=SessionStorageConfig(backend="redis"),
    )

    mcp = server.create_mcp(cfg)

    build_store.assert_called_once_with(cfg)
    assert create_auth.call_args.kwargs["client_storage"] is store
    assert mcp.auth is create_auth.return_value


@pytest.mark.parametrize("mode", ["passthrough", "kubernetes"])
def test_non_oidc_auth_does_not_build_oauth_storage(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_store = MagicMock(side_effect=AssertionError("OAuth storage is unused"))
    kubernetes_auth = MagicMock()
    monkeypatch.setattr(server, "_build_session_store", build_store)
    monkeypatch.setattr(server, "create_kubernetes_auth", kubernetes_auth)

    mcp = server.create_mcp(Config(auth=AuthConfig(mode=mode)))

    build_store.assert_not_called()
    if mode == "kubernetes":
        kubernetes_auth.assert_called_once_with()
        assert mcp.auth is kubernetes_auth.return_value
    else:
        kubernetes_auth.assert_not_called()
        assert mcp.auth is None


def test_http_tracing_preserves_health_and_isolates_server_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_tracer", None)
    cfg = Config(server=ServerConfig(transport="http"))
    first = server.create_mcp(cfg)
    second = server.create_mcp(cfg)

    @first.custom_route("/request-context", methods=["GET"])
    async def request_context(request: Request) -> JSONResponse:
        return JSONResponse({"request_id": request_id_var.get()})

    with TestClient(server._build_http_app(first, cfg)) as client:
        assert client.get("/health").json() == {
            "status": "healthy",
            "service": "mcp-server",
        }
        first_id = client.get("/request-context").json()["request_id"]
        second_id = client.get("/request-context").json()["request_id"]
        assert first_id and second_id and first_id != second_id

    assert request_id_var.get() is None
    with TestClient(server._build_http_app(second, cfg)) as client:
        assert client.get("/request-context").status_code == 404
