"""Task-level HTML dashboard for a running (or finished) rollout.

    uv run python -m sd.dashboard runs/train250     # then open http://127.0.0.1:8765

Serves sd/dashboard.html and the run's status.json (written by sd.harbor_rollouts). The page
refreshes itself every few seconds. Binds to 127.0.0.1 only.
"""

import http.server
from functools import partial
from pathlib import Path

import typer

PAGE = Path(__file__).with_name("dashboard.html")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, run_dir: Path, **kwargs):
        self.run_dir = run_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path.split("?")[0] == "/status.json":
            status = self.run_dir / "status.json"
            if status.exists():
                self._send(status.read_bytes(), "application/json")
            else:
                self.send_error(404, "no status.json yet")
        else:
            self.send_error(404)

    def _send(self, body: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the terminal quiet
        pass


def main(run_dir: Path, port: int = 8765) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, run_dir=run_dir))
    print(f"Dashboard for {run_dir}: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    typer.run(main)
