"""Msgpack WebSocket server for a stateful robot policy."""

from __future__ import annotations

import asyncio
import http
import logging
import ssl
import time
import traceback
from typing import Protocol

import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from lerobot.scripts.obm_inference import msgpack_numpy

LOGGER = logging.getLogger(__name__)


class Policy(Protocol):
    @property
    def metadata(self) -> dict: ...

    def step(self, observation: dict) -> dict: ...


class WebsocketPolicyServer:
    def __init__(
        self,
        policy: Policy,
        *,
        host: str = "0.0.0.0",
        port: int = 8999,
        ssl_certfile: str | None = None,
        ssl_keyfile: str | None = None,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.ssl_context: ssl.SSLContext | None = None
        if ssl_certfile:
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.load_cert_chain(ssl_certfile, ssl_keyfile)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ssl=self.ssl_context,
            process_request=_health_check,
        ) as server:
            LOGGER.info("Listening on %s:%d", self.host, self.port)
            await server.serve_forever()

    async def _handler(
        self,
        websocket: websocket_server.ServerConnection,
    ) -> None:
        LOGGER.info("Connection opened from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.policy.metadata))
        previous_total_time: float | None = None

        while True:
            try:
                started = time.monotonic()
                request = msgpack_numpy.unpackb(await websocket.recv())
                if not isinstance(request, dict):
                    raise TypeError(f"Request must be a dict, got {type(request).__name__}.")
                observation = request.get("obs", request)
                inference_started = time.monotonic()
                response = self.policy.step(observation)
                inference_ms = (time.monotonic() - inference_started) * 1000.0
                response["server_timing"] = {"infer_ms": inference_ms}
                if previous_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total_time * 1000.0
                await websocket.send(packer.pack(response))
                previous_total_time = time.monotonic() - started
            except websockets.ConnectionClosed:
                LOGGER.info(
                    "Connection closed from %s",
                    websocket.remote_address,
                )
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error; traceback sent in previous frame.",
                )
                raise


def _health_check(
    connection: websocket_server.ServerConnection,
    request: websocket_server.Request,
) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
