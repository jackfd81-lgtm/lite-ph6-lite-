import json
import urllib.request
import urllib.error
import subprocess
from .backend import active_backend, JETSON_HOST, JETSON_PORT

OLLAMA_LOCAL = "http://localhost:11434"
MODE_MODEL = {
    "chat":   "llama3.2",
    "code":   "qwen2.5-coder:1.5b",
    "reason": "qwen3:1.7b",
}


def ask(mode: str, prompt: str) -> dict:
    backend = active_backend()
    if backend == "jetson":
        return _ask_jetson(mode, prompt)
    return _ask_local(mode, prompt)


def _ask_jetson(mode: str, prompt: str) -> dict:
    url = f"http://{JETSON_HOST}:{JETSON_PORT}/{mode}"
    payload = json.dumps({"mode": mode, "prompt": prompt}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"status": "error", "backend": "jetson", "error": str(e),
                "output": _ask_local(mode, prompt)["output"]}


def _ask_local(mode: str, prompt: str) -> dict:
    model = MODE_MODEL.get(mode, "llama3.2")
    url = f"{OLLAMA_LOCAL}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return {"status": "ok", "backend": "local", "model": model,
                    "output": data.get("response", "")}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"status": "error", "backend": "local", "error": str(e), "output": ""}
