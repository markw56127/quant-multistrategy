"""Block until Yahoo's API host resolves (or a timeout passes). Jobs started during a brief
dark wake can run before the network is up, and then every download fails with
'curl: (6) Could not resolve host' (2026-10-13)."""

import socket
import time

from loguru import logger


def wait_for_network(host="query1.finance.yahoo.com", timeout=300, step=10):
    t0 = time.time()
    while True:
        try:
            socket.getaddrinfo(host, 443)
            return True
        except OSError:
            if time.time() - t0 > timeout:
                logger.error(f"network still down after {timeout}s; giving up this run")
                return False
            time.sleep(step)
