import json
import urllib.request
import urllib.error

OLLAMA_HOST = "http://localhost:11434"
MODE_MODEL = {
    "chat":   "llama3.2",
    "code":   "qwen2.5-coder:1.5b",
    "reason": "qwen3:1.7b",
}


def run(mode: str, prompt: str) -> dict:
    model = MODE_MODEL.get(mode, "llama3.2")
    url = f"{OLLAMA_HOST}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return {"status": "ok", "model": model, "output": data.get("response", "")}
    except urllib.error.URLError as e:
        return {"status": "error", "model": model, "error": str(e), "output": ""}
