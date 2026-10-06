"""HTTP front end of the console agent: auth, routing and JSON responses."""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from src.console_agent import status
from src.utils.logger import logger
from src.utils.timezone import tz

LOGIN_HEADER = "Tailscale-User-Login"
_STRATEGY_NAME = re.compile(r"[a-z][a-z0-9_]{0,15}")


class AgentHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        # tailscale serve sets this header for tailnet users; requests without it never came through serve.
        if self.headers.get(LOGIN_HEADER) != self.server.config.operator_login:
            self._send_json(403, {"error": "forbidden"})
            return
        url = urlsplit(self.path)
        try:
            if url.path == "/status":
                self._send_json(200, status.build_status(self.server.config, tz.now()))
            elif url.path == "/decisions":
                self._send_decision(parse_qs(url.query).get("strategy", [""])[0])
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as e:
            logger.error(f"[console-agent] {url.path} failed: {e!r}")
            self._send_json(500, {"error": "internal error"})

    def _method_not_allowed(self) -> None:
        self._send_json(405, {"error": "read-only agent"})

    do_POST = do_PUT = do_PATCH = do_DELETE = _method_not_allowed

    def _send_decision(self, strategy: str) -> None:
        if not _STRATEGY_NAME.fullmatch(strategy):
            self._send_json(400, {"error": "invalid strategy name"})
            return
        if strategy not in status.read_toggle(self.server.config.toggle_path):
            self._send_json(404, {"error": f"unknown strategy {strategy}"})
            return
        record = status.read_latest_decision(self.server.config.latest_dir, strategy)
        if record is None:
            self._send_json(404, {"error": f"no decision recorded for {strategy}"})
            return
        self._send_json(200, record)

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_request(self, code="-", size="-") -> None:
        pass  # the console polls every 10 s, so per-request lines would flood the journal

    def log_error(self, format: str, *args) -> None:
        logger.error(f"[console-agent] {format % args}")


class AgentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: status.AgentConfig):
        super().__init__(address, AgentHandler)
        self.config = config


def make_server(config: status.AgentConfig, host: str = "127.0.0.1", port: int = 8090) -> AgentServer:
    return AgentServer((host, port), config)
