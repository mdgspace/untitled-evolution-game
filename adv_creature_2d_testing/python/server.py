"""Bounded protocol-v3 TCP server for the Unity live ecosystem."""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import struct
import traceback
from typing import Any, Dict

import torch

from live_ecosystem import LiveEcosystem, PROTOCOL_VERSION

HOST = "127.0.0.1"
PORT = int(os.environ.get("LIVE_ECOSYSTEM_PORT", "9999"))
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
SOCKET_TIMEOUT_SECONDS = 5.0
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "live_ecosystem_v5.pt")


def archive_legacy_checkpoint() -> None:
    for version in (2, 3, 4):
        old = os.path.join(CHECKPOINT_DIR, f"live_ecosystem_v{version}.pt")
        legacy = os.path.join(CHECKPOINT_DIR, "legacy", f"live_ecosystem_v{version}.pt")
        if os.path.exists(old) and not os.path.exists(CHECKPOINT):
            os.makedirs(os.path.dirname(legacy), exist_ok=True)
            if not os.path.exists(legacy):
                shutil.move(old, legacy)
                print(f"[CHECKPOINT] archived incompatible v{version} at {legacy}", flush=True)


def recv_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data.extend(chunk)
    return bytes(data)


def recv_message(connection: socket.socket) -> Dict[str, Any]:
    size = struct.unpack(">I", recv_exact(connection, 4))[0]
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid message size {size}")
    value = json.loads(recv_exact(connection, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("message root must be an object")
    return value


def send_message(connection: socket.socket, message: Dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"response is too large: {len(payload)} bytes")
    connection.sendall(struct.pack(">I", len(payload)) + payload)


def main() -> None:
    # Unity's render/physics threads take priority in performance mode. One
    # Torch worker still trains continuously without competing for every core.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    archive_legacy_checkpoint()
    ecosystem = LiveEcosystem(CHECKPOINT)
    stopping = False

    def request_stop(*_args: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((HOST, PORT))
            listener.listen(1)
            listener.settimeout(1.0)
            print(f"[LIVE ECOSYSTEM] protocol {PROTOCOL_VERSION} listening on {HOST}:{PORT}", flush=True)
            while not stopping:
                try:
                    connection, address = listener.accept()
                except socket.timeout:
                    continue
                print(f"[LIVE ECOSYSTEM] Unity connected from {address}", flush=True)
                with connection:
                    connection.settimeout(SOCKET_TIMEOUT_SECONDS)
                    try:
                        send_message(connection, ecosystem.initial_response())
                        while not stopping:
                            request = recv_message(connection)
                            if request.get("kind") == "shutdown":
                                send_message(connection, {"protocol_version": PROTOCOL_VERSION,
                                                          "server_status": "Shutting down"})
                                stopping = True
                                break
                            send_message(connection, ecosystem.observe(request))
                    except (ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
                        print(f"[LIVE ECOSYSTEM] Unity disconnected: {exc}", flush=True)
                    except Exception as exc:
                        # A model/data error must not take down the long-lived
                        # server. Unity can reconnect after the offending
                        # response is discarded, while the traceback remains
                        # visible in the development HUD/log.
                        print(f"[LIVE ECOSYSTEM] session error: {exc}", flush=True)
                        traceback.print_exc()
    finally:
        ecosystem.shutdown()
        print("[LIVE ECOSYSTEM] checkpoint saved; stopped", flush=True)


if __name__ == "__main__":
    main()
