#!/usr/bin/env python3
# Tiny dependency-free TCP forwarder: expose Lite3 dashboard ports on pro6000's
# localhost so VS Code Remote-SSH auto-forwards them to the Mac.
import contextlib
import socket
import threading

# (listen_host, listen_port) -> (target_host, target_port)
MAP = [
    ("127.0.0.1", 8080, "192.168.1.120", 8080),  # dashboard
    ("127.0.0.1", 8888, "192.168.1.120", 8888),  # camera HLS (tunnel-safe)
]


def pipe(a, b):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except Exception:
        pass
    finally:
        for s in (a, b):
            with contextlib.suppress(Exception):
                s.shutdown(socket.SHUT_RDWR)


def handle(client, thost, tport):
    try:
        remote = socket.create_connection((thost, tport), timeout=5)
    except Exception:
        client.close()
        return
    threading.Thread(target=pipe, args=(client, remote), daemon=True).start()
    threading.Thread(target=pipe, args=(remote, client), daemon=True).start()


def listener(lhost, lport, thost, tport):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((lhost, lport))
    s.listen(64)
    print(f"forward {lhost}:{lport} -> {thost}:{tport}", flush=True)
    while True:
        c, _ = s.accept()
        threading.Thread(target=handle, args=(c, thost, tport), daemon=True).start()


def main():
    for lhost, lport, thost, tport in MAP:
        threading.Thread(
            target=listener, args=(lhost, lport, thost, tport), daemon=True
        ).start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
