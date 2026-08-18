#!/usr/bin/env python3
# Tiny dependency-free TCP forwarder: expose Lite3 dashboard ports on pro6000's
# localhost so VS Code Remote-SSH auto-forwards them to the Mac.
import contextlib
import socket
import threading

# (listen_host, listen_port) -> (target_hosts, target_port)
#
# Two target addresses, tried in order, because the robot has two and the wired one is
# the one that disappears. eth1 is 192.168.1.120; wlan0 is 10.42.0.191 on the SyncAI-WiFi
# network pro6000 itself hosts. Unplugging the cable used to take the dashboard down even
# though the robot stayed fully reachable over wifi and ssh -- the target was a single
# hard-coded address, so nothing here could notice the other route existed.
#
# Wired first: lower latency, and it does not depend on the AP staying up. The fallback
# costs one refused connection, which is sub-millisecond on a LAN.
ROBOT_HOSTS = ("192.168.1.120", "10.42.0.191")
MAP = [
    ("127.0.0.1", 8080, ROBOT_HOSTS, 8080),  # dashboard
    ("127.0.0.1", 8888, ROBOT_HOSTS, 8888),  # camera HLS (tunnel-safe)
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


def _connect(thosts, tport):
    """First address that answers. Returns None if none do."""
    for host in thosts:
        try:
            return socket.create_connection((host, tport), timeout=2)
        except OSError:
            continue
    return None


def handle(client, thosts, tport):
    remote = _connect(thosts, tport)
    if remote is None:
        client.close()
        return
    threading.Thread(target=pipe, args=(client, remote), daemon=True).start()
    threading.Thread(target=pipe, args=(remote, client), daemon=True).start()


def listener(lhost, lport, thosts, tport):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((lhost, lport))
    s.listen(64)
    print(f"forward {lhost}:{lport} -> {'|'.join(thosts)}:{tport}", flush=True)
    while True:
        c, _ = s.accept()
        threading.Thread(target=handle, args=(c, thosts, tport), daemon=True).start()


def main():
    for lhost, lport, thosts, tport in MAP:
        threading.Thread(
            target=listener, args=(lhost, lport, thosts, tport), daemon=True
        ).start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
