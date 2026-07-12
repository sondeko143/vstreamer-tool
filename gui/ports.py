import socket


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def allocate_free_port(claimed: set[int], base: int = 8080, limit: int = 65535) -> int:
    port = base
    while port <= limit:
        if port not in claimed and is_port_free(port):
            return port
        port += 1
    raise RuntimeError(f"no free port available from {base}")
