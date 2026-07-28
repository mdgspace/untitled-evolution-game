import socket
import struct
import json

HOST = "127.0.0.1"
PORT = 9999


def recv_exact(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        data += chunk
    return data


def recv_message(conn):
    # 4-byte big-endian length prefix, then that many bytes of UTF-8 JSON.
    # TCP is a byte stream, not a message stream -- without this framing,
    # there's no reliable way to know where one JSON payload ends and the
    # next begins.
    length_bytes = recv_exact(conn, 4)
    (length,) = struct.unpack(">I", length_bytes)
    payload = recv_exact(conn, length)
    return json.loads(payload.decode("utf-8"))


def send_message(conn, obj):
    payload = json.dumps(obj).encode("utf-8")
    header = struct.pack(">I", len(payload))
    conn.sendall(header + payload)


def compute_outputs(inputs_message):
    # PLACEHOLDER -- this is the ONLY function where real brain logic
    # eventually goes. Right now it just echoes zero deltas for every
    # limb it was sent, so you can verify the whole loop (Unity blocks
    # correctly, waits, receives, applies) before any real decision-making
    # exists -- same spirit as the GenerateMockOutputs() stub on the
    # Unity side, just on this end of the wire instead.
    response = {"creatures": {}}
    for creature_id, creature_data in inputs_message["creatures"].items():
        deltas = {limb_id: 0.0 for limb_id in creature_data["local_inputs"].keys()}
        response["creatures"][creature_id] = {"deltas": deltas}
    return response


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Waiting for Unity to connect on {HOST}:{PORT}...")
    conn, addr = server.accept()
    print(f"Unity connected from {addr}")

    try:
        while True:
            inputs_message = recv_message(conn)
            outputs_message = compute_outputs(inputs_message)
            send_message(conn, outputs_message)
    except ConnectionError:
        print("Unity disconnected.")
    finally:
        conn.close()
        server.close()


if __name__ == "__main__":
    main()