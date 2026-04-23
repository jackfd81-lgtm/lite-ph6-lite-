import socket
import os

JETSON_HOST = os.environ.get("PH6_JETSON_HOST", "")
JETSON_PORT = int(os.environ.get("PH6_JETSON_PORT", 8765))
TIMEOUT = 2


def jetson_reachable() -> bool:
    if not JETSON_HOST:
        return False
    try:
        with socket.create_connection((JETSON_HOST, JETSON_PORT), timeout=TIMEOUT):
            return True
    except OSError:
        return False


def active_backend() -> str:
    return "jetson" if jetson_reachable() else "local"
