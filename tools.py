"""Tools for GAIA Level-1 agent. All free except LLM/transcription (via LiteLLM)."""

from __future__ import annotations
import io
import os
import re
import contextlib
import requests
from pathlib import Path

API_BASE = os.getenv("GAIA_API_BASE", "https://agents-course-unit4-scoring.hf.space")
DOWNLOAD_DIR = Path(os.getenv("GAIA_DOWNLOAD_DIR", "./downloads"))
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

_HEADERS = {
    "User-Agent": "hhf-gaia-agent/1.0 (mailto:student@example.com) educational-project"
}


def web_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo text search, no key needed (with retry for rate limits)."""
    import time

    last = "unknown"
    for attempt in range(3):
        try:
            from ddgs import DDGS

            out = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    out.append(f"- {r.get('title')}: {r.get('body')} ({r.get('href')})")
            if out:
                return "\n".join(out)
            last = "No results."
        except Exception as e:
            last = str(e)
            time.sleep(1.5 * (attempt + 1))
    return f"web_search error: {last}"


def fetch_wiki_page(title: str, max_chars: int = 8000) -> str:
    """Full Wikipedia article text via MediaWiki extracts API (no scraping)."""
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": title,
                "format": "json",
            },
            headers=_HEADERS,
            timeout=20,
        ).json()
        pages = r.get("query", {}).get("pages", {})
        text = next(iter(pages.values()), {}).get("extract", "")
        return text[:max_chars] or f"No extract for '{title}'."
    except Exception as e:
        return f"fetch_wiki_page error: {e}"


def fetch_page(url: str, max_chars: int = 8000) -> str:
    """Fetch page and return readable text (Wikipedia URLs go via API extracts)."""
    try:
        m = re.search(r"wikipedia\.org/wiki/([^#?]+)", url)
        if m:
            import urllib.parse

            return fetch_wiki_page(
                urllib.parse.unquote(m.group(1).replace("_", " ")), max_chars
            )
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(r.text, "html.parser")
            for t in soup(["script", "style", "nav", "footer"]):
                t.decompose()
            text = soup.get_text(separator="\n")
        except Exception:
            text = r.text
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text[:max_chars]
    except Exception as e:
        return f"fetch_page error: {e}"


def wiki_search(query: str) -> str:
    """MediaWiki API search + summary, no key needed."""
    try:
        s = requests.Session()
        s.headers.update(_HEADERS)
        sr = s.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 3,
            },
            timeout=20,
        ).json()
        hits = sr.get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia hits."
        lines = []
        for h in hits:
            title = h["title"]
            try:
                summ = (
                    s.get(
                        "https://en.wikipedia.org/api/rest_v1/page/summary/"
                        + title.replace(" ", "_"),
                        timeout=20,
                    )
                    .json()
                    .get("extract", "")
                )
            except Exception:
                summ = h.get("snippet", "")
            lines.append(f"## {title}\n{summ[:1500]}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"wiki_search error: {e}"


def youtube_transcript(url_or_id: str) -> str:
    """YouTube captions via youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url_or_id)
        vid = m.group(1) if m else url_or_id.strip()
        api = YouTubeTranscriptApi()
        tracks = api.list(vid)
        tr = next(iter(tracks), None)
        if tr is None:
            return "No transcripts."
        chunks = tr.fetch() if hasattr(tr, "fetch") else api.fetch(vid)
        texts = [c.text if hasattr(c, "text") else c.get("text", "") for c in chunks]
        return " ".join(texts)[:12000]
    except Exception as e:
        return f"youtube_transcript error: {e}"


def get_task_file(task_id: str, file_name: str = "") -> str:
    """Download /files/{task_id}; on 404 fallback to GAIA HF dataset cache. Returns local path or error."""
    dest = DOWNLOAD_DIR / (file_name or task_id)
    if dest.exists():
        return str(dest)
    # 1) scoring API
    try:
        r = requests.get(f"{API_BASE}/files/{task_id}", timeout=30)
        if (
            r.status_code == 200
            and "task_id" not in str(r.headers.get("content-type", ""))
            or (
                r.status_code == 200
                and len(r.content) > 200
                and b"detail" not in r.content[:100]
            )
        ):
            # crude check: real file vs {"detail":...}
            if not (r.content[:50].strip().startswith(b'{"detail"')):
                dest.write_bytes(r.content)
                return str(dest)
    except Exception:
        pass
    # 2) fallback: gated GAIA repo via hf_hub_download (needs HF_TOKEN + accepted conditions)
    try:
        from datasets import load_dataset

        token = os.getenv("HF_TOKEN")
        ds = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_level1",
            split="validation",
            token=token or None,
        )
        row = next((x for x in ds if str(x.get("task_id")) == task_id), None)
        if row:
            fp = (row.get("file_path") or "").strip()
            if fp and os.path.isabs(fp) and os.path.exists(fp):
                import shutil

                shutil.copy(fp, dest)
                return str(dest)
            if fp:
                # relative path inside repo, e.g. 2023/validation/xxx.png
                try:
                    from huggingface_hub import hf_hub_download

                    if not token:
                        return (
                            f"No file for {task_id}: gated repo needs HF_TOKEN in .env "
                            f"(accept conditions at https://huggingface.co/datasets/gaia-benchmark/GAIA). "
                            f"file_name={file_name}"
                        )
                    dl = hf_hub_download(
                        repo_id="gaia-benchmark/GAIA",
                        filename=fp,
                        repo_type="dataset",
                        token=token,
                    )
                    import shutil

                    shutil.copy(dl, dest)
                    return str(dest)
                except Exception as e:
                    return (
                        f"No file for {task_id}: hub download failed ({str(e)[:200]}). "
                        f"Check HF_TOKEN + accepted gating. file_name={file_name}"
                    )
        return f"No file for {task_id} (API 404, dataset has no cached file). file_name={file_name}"
    except Exception as e:
        return f"get_task_file error: {e}. file_name={file_name}"


def run_python(code: str, timeout_note: str = "") -> str:
    """Execute python snippet (pandas/openpyxl available), capture stdout. For .py/.xlsx/math."""
    buf = io.StringIO()
    g = {"__name__": "__tool__"}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, g)
        out = buf.getvalue()
        if "result" in g and not out.strip():
            out = str(g["result"])
        return (
            out.strip()
            or f"OK (no output). vars: {[k for k in g if not k.startswith('_')][:10]}"
        )[:6000]
    except Exception as e:
        return f"run_python error: {e}"


def _router_kwargs() -> dict:
    kw = {}
    base = os.getenv("LITELLM_API_BASE") or os.getenv("OPENAI_API_BASE")
    key = os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if base:
        kw["api_base"] = base
    if key:
        kw["api_key"] = key
    return kw


def _normalize_model_name(m: str) -> str:
    base = os.getenv("LITELLM_API_BASE") or os.getenv("OPENAI_API_BASE")
    if "/" not in m and base:
        return f"openai/{m}"
    return m


def transcribe_audio(path: str) -> str:
    """Audio -> text: 1) MEDIA_MODEL chat completion with audio input, 2) local faster-whisper fallback."""
    import base64

    media_model = _normalize_model_name(
        os.getenv("MEDIA_MODEL") or os.getenv("MODEL", "gpt-4o-mini")
    )
    # 1) multimodal chat model (gemma4 with audio decoder) via input_audio
    try:
        import litellm

        with open(path, "rb") as f:
            raw = f.read()
        # cap ~2MB to spare slow NAS inference
        raw = raw[:2_000_000]
        b64 = base64.b64encode(raw).decode()
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "mp3"
        fmt = "wav" if ext not in ("mp3", "wav") else ext
        r = litellm.completion(
            model=media_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe this audio verbatim. Return only the transcript.",
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": fmt},
                        },
                    ],
                }
            ],
            max_tokens=2000,
            temperature=0,
            timeout=180,
            **_router_kwargs(),
        )
        return (r.choices[0].message.content or "").strip()[:8000]
    except Exception as e:
        last_err = str(e)[:300]
    # fallback: local faster-whisper (pip install faster-whisper), no key needed
    try:
        from faster_whisper import WhisperModel

        m = WhisperModel(
            os.getenv("WHISPER_LOCAL", "tiny"), device="cpu", compute_type="int8"
        )
        segs, _ = m.transcribe(path)
        return " ".join(s.text for s in segs).strip()[:8000] or "(empty transcript)"
    except ImportError:
        return f"No STT: media model '{media_model}' failed ({last_err}). pip install faster-whisper for local fallback. File at {path}"
    except Exception as e:
        return f"local whisper failed: {e} (router err: {last_err}). File at {path}"


def analyze_image(path: str, question: str = "") -> str:
    """Image -> description via LiteLLM vision model (needs vision-capable MODEL)."""
    try:
        import base64, litellm

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question or "Describe this image precisely.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ]
        r = litellm.completion(
            model=_normalize_model_name(
                os.getenv("MEDIA_MODEL") or os.getenv("MODEL", "gpt-4o-mini")
            ),
            messages=msgs,
            max_tokens=1000,
            timeout=180,
            **_router_kwargs(),
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"analyze_image needs vision MODEL + key: {e}. File at {path}"


# --- LiteLLM function schemas ---
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "DuckDuckGo web search for facts, menus, awards, articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch URL text (articles, LibreTexts, Universe Today).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "English Wikipedia search (albums, dinosaurs, Olympics, pitchers).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_wiki_page",
            "description": "Get FULL Wikipedia article text by title (use for discography, filmography, medal tables). Prefer over fetch_page for wikipedia.org.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_transcript",
            "description": "Get YouTube captions for bird-count / quote questions.",
            "parameters": {
                "type": "object",
                "properties": {"url_or_id": {"type": "string"}},
                "required": ["url_or_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_task_file",
            "description": "Download GAIA attachment by task_id (png/mp3/py/xlsx). Returns local path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "file_name": {"type": "string", "default": ""},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run python for math, .py files, .xlsx totals (pandas/openpyxl present).",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe mp3 via MEDIA_MODEL multimodal chat (gemma4 audio decoder).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Describe png attachment via vision model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "question": {"type": "string", "default": ""},
                },
                "required": ["path"],
            },
        },
    },
]

DISPATCH = {
    "web_search": web_search,
    "fetch_page": fetch_page,
    "fetch_wiki_page": fetch_wiki_page,
    "wiki_search": wiki_search,
    "youtube_transcript": youtube_transcript,
    "get_task_file": get_task_file,
    "run_python": run_python,
    "transcribe_audio": transcribe_audio,
    "analyze_image": analyze_image,
}
