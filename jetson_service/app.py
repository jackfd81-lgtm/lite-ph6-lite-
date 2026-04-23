from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from ollama_runner import run

PORT = 8765


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        mode = self.path.lstrip("/")
        if mode not in ("chat", "code", "reason"):
            self._respond(400, {"error": f"unknown mode: {mode}"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        prompt = body.get("prompt", "")
        result = run(mode, prompt)
        self._respond(200, result)

    def _respond(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # suppress default access log


if __name__ == "__main__":
    print(f"Jetson advisory service on port {PORT}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
