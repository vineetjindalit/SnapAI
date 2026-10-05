"""
models/vlm_curator.py — OPTIONAL photographer-grade album judge (VLM), used
ONLY at album time (never in the real-time capture loop).

Design rules (so it never makes the system hotch-potch or fragile):
  • OFF BY DEFAULT. With no credentials it is a pure no-op (`available == False`)
    and the album falls back to the on-device classifier — nothing breaks.
  • ISOLATED. One module, one call site (api/routes.album). The rest of the
    system doesn't know or care whether a VLM exists.
  • DEFENSIVE. Any network/parse error → returns None → graceful fallback.
  • PRIVACY-EXPLICIT. It uploads the FEW already-captured album photos (not the
    live stream) to a cloud endpoint, and only when the user has opted in by
    setting the env vars below.

Enable (OpenAI-compatible vision endpoint — OpenAI, OpenRouter, local llama.cpp
server, etc.):
    export SNAPPY_VLM_API_KEY=...           # required to turn it on
    export SNAPPY_VLM_URL=https://api.openai.com/v1/chat/completions   # default
    export SNAPPY_VLM_MODEL=gpt-4o-mini     # any vision model the endpoint serves

Returns, per photo URL: {"moment": <taxonomy label>, "keep": bool, "caption": str}.
"""
from __future__ import annotations
import os, json, base64, urllib.request, urllib.error
from typing import Dict, List, Optional

# The birthday taxonomy the VLM must label within (kept identical to training so
# album tags stay consistent with the on-device model).
TAXONOMY = ["pre_preparation", "person_arrival", "surprise_celebration", "cake",
            "cake_person", "cake_with_candles", "candle_blowing", "cake_cutting",
            "cake_smashing", "cake_feeding", "birthday_gifting", "group_photo",
            "individual_people", "smiling_moments", "laughing_moments",
            "crying_moments", "hugging_moments", "dancing_moments",
            "gazing_moments", "not_birthday"]

_MAX_PHOTOS = int(os.environ.get("SNAPPY_VLM_MAX_PHOTOS", "60"))   # cost guard
_TIMEOUT    = float(os.environ.get("SNAPPY_VLM_TIMEOUT", "30"))


class VLMCurator:
    _inst = None

    def __init__(self):
        self.key   = os.environ.get("SNAPPY_VLM_API_KEY", "").strip()
        self.url   = os.environ.get("SNAPPY_VLM_URL", "https://api.openai.com/v1/chat/completions")
        self.model = os.environ.get("SNAPPY_VLM_MODEL", "gpt-4o-mini")

    @classmethod
    def get(cls) -> "VLMCurator":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    @property
    def available(self) -> bool:
        return bool(self.key)

    def refine_album(self, photos: List[dict]) -> Optional[Dict[str, dict]]:
        """photos: [{url, filepath, moment_type}]. Returns {url: {moment, keep,
        caption}} or None on any problem (caller then keeps on-device tags)."""
        if not self.available or not photos:
            return None
        out: Dict[str, dict] = {}
        for p in photos[:_MAX_PHOTOS]:
            try:
                b = self._read_b64(p.get("filepath"))
                if b is None:
                    continue
                res = self._ask(b)
                if res:
                    out[p["url"]] = res
            except Exception:
                continue          # one photo failing must never sink the album
        return out or None

    # ── internals ────────────────────────────────────────────────────────────
    @staticmethod
    def _read_b64(path: Optional[str]) -> Optional[str]:
        if not path or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def _ask(self, b64: str) -> Optional[dict]:
        prompt = (
            "You are an expert birthday-event photographer curating an album. "
            "Look at this single photo and reply with STRICT JSON only: "
            '{"moment": one of ' + json.dumps(TAXONOMY) + ", "
            '"keep": true/false (true if it is a meaningful, well-composed birthday '
            "moment worth the album; false if blurry, redundant, or not a birthday), "
            '"caption": a short human caption}.'
        )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            "max_tokens": 200, "temperature": 0,
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
        txt = data["choices"][0]["message"]["content"]
        txt = txt[txt.find("{"): txt.rfind("}") + 1]      # tolerate ```json fences
        j = json.loads(txt)
        moment = j.get("moment")
        if moment not in TAXONOMY:
            moment = None
        return {"moment": moment, "keep": bool(j.get("keep", True)),
                "caption": str(j.get("caption", ""))[:160]}
