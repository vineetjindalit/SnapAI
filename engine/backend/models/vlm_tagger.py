"""
models/vlm_tagger.py — OPTIONAL photographer-grade album tagging via a
vision-language model (Qwen2.5-VL), at ALBUM time only (never real-time).

A VLM understands a birthday photo far better than our small classifier
("she's blowing the candles" vs "cutting the cake"), but it's seconds/photo and
multi-GB — fine once, at album build, not for live capture. So this runs ONLY
when generating the album, and ONLY when explicitly enabled.

Contract (kept deliberately clean so the codebase isn't hotch-potch):
  • ON by default (model already on disk). Disable with SNAPPY_VLM=0 (model: SNAPPY_VLM_MODEL,
    default Qwen/Qwen2.5-VL-3B-Instruct — downloaded once on first use).
  • Lazy-loads. If transformers/model/hardware aren't ready → .ok = False and
    every call returns None, so the album falls back to the on-device
    classifier and ALWAYS builds. Never raises into the pipeline.
  • Pure tagger: image → one birthday-moment label. No state, no side effects.
"""
from __future__ import annotations
import json
import os
import re
import threading
import time
import logging

log = logging.getLogger("snappy.vlm")

# The signals the backend actually has available per live frame — the intent
# parser is told to pick ONLY from this set so its output always maps onto a
# real detector (models/gaze_estimator.py, models/hsemotion_fer.py,
# models/moment_predictor.py's motion diff, and frame-brightness). "clip" is
# always implied and doesn't need to be listed explicitly.
_KNOWN_DETECTORS = ("clip", "face", "emotion", "gaze", "motion", "lighting")

_INTENT_SYSTEM_PROMPT = """You are SnapAI, a friendly AI camera assistant. A \
user just told you what moment they want you to capture. Translate their \
request into a visual detection objective and a short, warm, natural reply.

Respond with STRICT JSON only, no markdown fences, matching exactly this \
shape:
{
  "understood": true,
  "reply": "<one warm, natural sentence acknowledging the specific request, \
varied wording, never robotic — e.g. 'Got it! Strike your pose when you're \
ready — I'll capture it automatically.'>",
  "watching_for": "<short 2-6 word label, e.g. 'a pose'>",
  "guide": "<one short instruction telling the user what to do now, e.g. \
'Strike your pose when you're ready.'>",
  "clip_prompt": "<a short, visually concrete English sentence describing \
EXACTLY what the camera should see when the moment happens, written for an \
image-matching model — e.g. 'a person posing for a photograph' or 'a bright, \
well-lit room with the lights on'>",
  "detectors": [<subset of "face","emotion","gaze","motion","lighting" that \
genuinely helps confirm this specific moment; omit ones that don't apply>],
  "capture_mode": "<'one_time' if they want ONE capture, 'continuous' if they \
said 'every time'/'whenever'/'keep watching', 'stop' if the ENTIRE request is \
about CANCELLING/ENDING/STOPPING current watching (starts with or means \
'stop'/'never mind'/'cancel'/'that's enough'/'don't watch for X anymore' — \
in that case set watching_for/guide/clip_prompt/detectors to empty values, \
they are irrelevant), 'replace' otherwise>",
  "needs_clarification": <true ONLY if the request is genuinely too vague to \
act on — e.g. 'capture the special moment' with no hint of what to look \
for>,
  "clarify_question": "<if needs_clarification, one short question to ask; \
otherwise null>"
}

Act immediately whenever the request is reasonably clear, even if casually \
phrased — do not over-ask for clarification. Only ask when you genuinely \
cannot form a visual objective from the request. Keep every string SHORT —
one sentence max each. Output ONLY the JSON object, nothing before or after."""

# General Mode: given ONLY an event/occasion name (no pre-built moment
# taxonomy like birthday has, no per-moment prompting like Custom Mode
# needs), derive what a good photographer would make sure to capture —
# drawing on the model's general world knowledge of the occasion's typical
# customs, entirely at request time.
#
# Two things this is tuned for beyond just "get valid JSON back":
#   1. GENERALIZE past the handful of occasions actually tested (Diwali,
#      Rakhi) to arbitrary events — cultural/religious festivals from any
#      tradition, family rituals, weddings, sports, corporate, school, or an
#      occasion the model has never specifically heard of. The DIVERSITY
#      instruction + explicit unfamiliar-event fallback below are what make
#      that hold up beyond the two occasions used to build this.
#   2. SPEED: every extra word asked for here is an extra generated token,
#      and this call runs on the live critical path the user is staring at
#      a screen waiting for (unlike VLM album tagging, which runs once in
#      the background). Moment count and field lengths are kept tight on
#      purpose — see derive_event_moments()'s max_new_tokens for the
#      matching budget.
_EVENT_MOMENTS_SYSTEM_PROMPT = """You are SnapAI's event-moment planner. A \
user just told you the name of an event or occasion — it could be ANYTHING: \
a cultural or religious festival from any tradition (e.g. "Diwali", "Eid", \
"Rakhi", "Hanukkah", "Lunar New Year"), a family ritual ("a naming \
ceremony", "a retirement party"), a life-stage event ("a wedding", "a baby \
shower", "a graduation"), or something ordinary ("a team offsite", "my \
son's soccer game", "book club"). List the 4 to 6 most important, visually \
DISTINCT, photographable moments a good photographer would make sure not \
to miss.

Make the moments span DIFFERENT KINDS of content, not variations on the \
same shot — mix across: a signature ritual/action specific to this occasion \
(if one exists), a social/group moment (people together), an emotional or \
candid moment (reactions, expressions), and an object/setting moment (food, \
decor, a key prop) where relevant. Distinct moments matter because they get \
told apart from EACH OTHER by an image-matching model — near-duplicate \
entries are actively harmful, not just redundant.

If the occasion is unfamiliar, generic, or has no strong cultural script \
(e.g. "a team meeting", "my nephew's soccer game"), do NOT leave the list \
thin — fall back to the moments any real event of that general shape has: \
people arriving/gathering, the main activity or interaction happening, a \
group photo, and visible celebration or emotional reactions. Use your best \
judgment; never refuse just because the occasion is niche or ordinary.

Respond with STRICT JSON only, no markdown fences, matching exactly this \
shape:
{
  "understood": true,
  "event_summary": "<3-8 words: what kind of event this is>",
  "moments": [
    {
      "label": "<short human name, 2-4 words, e.g. 'lighting diyas'>",
      "clip_prompt": "<10-15 words: what a CAMERA would literally SEE \
during this moment, written for an image-matching model — describe the \
SCENE, not the occasion name, e.g. 'a person lighting small oil lamps at \
dusk', NOT 'Diwali lights'>",
      "detectors": [<subset of "face","emotion","gaze","motion","lighting" \
that plausibly helps confirm THIS moment; omit ones that don't apply>]
    }
  ]
}

Only set "understood": false and "moments": [] if the input is truly not an \
event at all (random gibberish) — a niche, ordinary, or unfamiliar occasion \
still gets a real answer via the fallback above. Keep every string to \
EXACTLY the length asked — this is a live, time-sensitive request. Output \
ONLY the JSON object, nothing before or after."""

# The label space the VLM must choose from (your birthday taxonomy).
VOCAB = [
    "candle_blowing", "cake_cutting", "cake_feeding", "cake_smashing",
    "cake_with_candles", "cake", "cake_person", "birthday_gifting",
    "group_photo", "surprise_celebration", "person_arrival", "pre_preparation",
    "dancing_moments", "hugging_moments", "laughing_moments", "crying_moments",
    "gazing_moments", "individual_people", "not_birthday",
]


class VLMTagger:
    _inst = None
    _lock = threading.Lock()

    def __init__(self):
        self.ok = False
        self._model = self._proc = self._torch = None
        self._dev = "cpu"
        if os.environ.get("SNAPPY_VLM", "1") != "1":
            log.info("VLM album tagger OFF (SNAPPY_VLM=0)")
            return
        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            name = os.environ.get("SNAPPY_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
            log.info(f"VLM album tagger: loading {name} (one-time download on first run)…")
            m = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16)
            self._dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = m.to(self._dev).eval()
            self._proc = AutoProcessor.from_pretrained(name)
            self._torch = torch
            self.ok = True
            log.info(f"VLM album tagger ready ({name} on {self._dev})")
        except Exception as e:
            self.ok = False
            log.info(f"VLM album tagger unavailable ({e}) — classifier will tag instead")

    @classmethod
    def get(cls):
        # Custom Mode's session-creation pre-warm (api/routes.py) and a live
        # prompt's actual parse_capture_intent() call can legitimately race
        # to construct this on two different threads at once — measured: the
        # unlocked check-then-create used to let BOTH threads start loading
        # the ~6GB model onto MPS concurrently, which doesn't just waste ~20s
        # twice, it can outright corrupt one of the loads (observed: "Cannot
        # copy out of meta tensor" from a torch device-transfer racing
        # itself). Mirrors CLIPEngine.get()'s identical lock for the exact
        # same reason.
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    def occasion(self, images_bgr) -> "str | None":
        """CONTEXT pass — look at several album photos TOGETHER and name the
        occasion ('birthday party', 'romantic celebration', …). Frame-level
        models can't see that a cake is being used for a date, not a birthday;
        the set of photos as a whole can. Album-time only. None on any failure.
        """
        if not self.ok or not images_bgr:
            return None
        try:
            import cv2
            from PIL import Image
            from qwen_vl_utils import process_vision_info
            content = []
            for im in images_bgr[:5]:
                if im is None:
                    continue
                h, w = im.shape[:2]
                m = max(h, w)
                if m > 448:
                    s = 448.0 / m
                    im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
                content.append({"type": "image",
                                "image": Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))})
            if not content:
                return None
            content.append({"type": "text", "text": (
                "These photos are from ONE filmed session. What is the occasion? "
                "Look at WHO is there and the SETTING, not just the props: a "
                "couple alone with wine, roses or petals is a romantic date or "
                "anniversary even if a cake appears; balloons, a 'Happy "
                "Birthday' banner, candles on a cake, or a group singing means "
                "a birthday party. Answer in 2-5 words only (e.g. 'birthday "
                "party', 'romantic date / anniversary', 'wedding', "
                "'casual get-together').")})
            msgs = [{"role": "user", "content": content}]
            text = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inp = self._proc(text=[text], images=imgs, videos=vids,
                             padding=True, return_tensors="pt").to(self._dev)
            with self._torch.no_grad():
                out = self._model.generate(**inp, max_new_tokens=16, do_sample=False)
            ans = self._proc.batch_decode(
                [o[len(i):] for i, o in zip(inp.input_ids, out)],
                skip_special_tokens=True)[0].strip()
            return ans[:80] or None
        except Exception as e:
            log.debug(f"VLM occasion failed: {e}")
            return None

    def is_celebration(self, images_bgr) -> "bool | None":
        """Binary context check for the album VETO — yes/no is far more robust
        from a 3B VLM than open-ended text. None = unsure → caller must NOT veto.
        """
        if not self.ok or not images_bgr:
            return None
        try:
            import cv2
            from PIL import Image
            from qwen_vl_utils import process_vision_info
            content = []
            for im in images_bgr[:4]:
                if im is None:
                    continue
                h, w = im.shape[:2]; m = max(h, w)
                if m > 448:
                    s = 448.0 / m
                    im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
                content.append({"type": "image",
                                "image": Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))})
            if not content:
                return None
            content.append({"type": "text", "text": (
                "Are these photos from a celebration event (birthday, party, "
                "wedding, anniversary — cake, gifts, decorations, toasts)? "
                "Answer with exactly one word: yes or no.")})
            msgs = [{"role": "user", "content": content}]
            text = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inp = self._proc(text=[text], images=imgs, videos=vids,
                             padding=True, return_tensors="pt").to(self._dev)
            with self._torch.no_grad():
                out = self._model.generate(**inp, max_new_tokens=3, do_sample=False)
            ans = self._proc.batch_decode(
                [o[len(i):] for i, o in zip(inp.input_ids, out)],
                skip_special_tokens=True)[0].strip().lower()
            if ans.startswith("yes"): return True
            if ans.startswith("no"):  return False
            return None
        except Exception as e:
            log.debug(f"VLM is_celebration failed: {e}")
            return None

    def is_birthday(self, image_bgr) -> "bool | None":
        """Strict binary check: does this photo show a birthday celebration
        (cake / candles / gifts / party)? Binary questions are far more
        reliable for VLMs than the 19-way forced choice — used to gate the
        zero-capture rescue so junk stays unrescued. None on failure.
        """
        if not self.ok or image_bgr is None:
            return None
        try:
            import cv2
            from PIL import Image
            from qwen_vl_utils import process_vision_info
            h, w = image_bgr.shape[:2]
            m = max(h, w)
            if m > 512:
                s = 512.0 / m
                image_bgr = cv2.resize(image_bgr, (max(1, int(w * s)), max(1, int(h * s))))
            pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": (
                    "Look carefully at everything, including any text visible on "
                    "screens, banners or cakes. Does this photo show a birthday "
                    "celebration — a birthday cake, candles, gifts, balloons, or "
                    "a Happy Birthday message anywhere? Answer strictly YES or NO.")}]}]
            text = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inp = self._proc(text=[text], images=imgs, videos=vids,
                             padding=True, return_tensors="pt").to(self._dev)
            with self._torch.no_grad():
                out = self._model.generate(**inp, max_new_tokens=4, do_sample=False)
            ans = self._proc.batch_decode(
                [o[len(i):] for i, o in zip(inp.input_ids, out)],
                skip_special_tokens=True)[0].strip().upper()
            if ans.startswith("YES"):
                return True
            if ans.startswith("NO"):
                return False
            return None
        except Exception as e:
            log.debug(f"VLM is_birthday failed: {e}")
            return None

    def tag(self, image_bgr):
        """Return one VOCAB label for a BGR image, or None (→ caller falls back)."""
        if not self.ok or image_bgr is None:
            return None
        try:
            import cv2
            from PIL import Image
            from qwen_vl_utils import process_vision_info
            # Downscale first — Qwen's latency scales with image tokens, so a 4K
            # frame is ~10x slower than a 512px one for zero tagging benefit.
            h, w = image_bgr.shape[:2]
            m = max(h, w)
            if m > 512:
                s = 512.0 / m
                image_bgr = cv2.resize(image_bgr, (max(1, int(w * s)), max(1, int(h * s))))
            pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            vocab = ", ".join(VOCAB)
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": (
                    "This is one photo from a birthday celebration. Identify the single "
                    "moment it shows. Reply with EXACTLY one label from this list and "
                    f"nothing else: {vocab}.")}]}]
            text = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inp = self._proc(text=[text], images=imgs, videos=vids,
                             padding=True, return_tensors="pt").to(self._dev)
            with self._torch.no_grad():
                out = self._model.generate(**inp, max_new_tokens=12, do_sample=False)
            ans = self._proc.batch_decode(
                [o[len(i):] for i, o in zip(inp.input_ids, out)],
                skip_special_tokens=True)[0].strip().lower()
            for tok in VOCAB:                       # snap the answer to a known label
                if tok in ans:
                    return tok
            return None
        except Exception as e:
            log.debug(f"VLM tag failed: {e}")
            return None

    def parse_capture_intent(self, user_text: str) -> "dict | None":
        """Custom Mode's NLU step: turn a free-text/voice capture request
        ("capture me when I pose") into a structured objective + a natural
        conversational reply, in ONE generation call.

        TEXT-ONLY (no image) — same loaded model as tag()/occasion() above,
        just a different capability, so this never costs a second model load.
        Measured 4.6-5.7s on this Mac (mps) — dominated by generation length,
        not a fixed per-call cost (confirmed: proportional to max_new_tokens).
        Real conversational-turn latency, not per-frame — but the CALLER
        still MUST dispatch it off the asyncio event loop (see
        api/server.py's _dispatch_custom_prompt) since it's a plain blocking
        call, like every other method on this class, and would otherwise
        stall every other session's frames for the duration.

        Returns the parsed dict, or None on ANY failure (VLM unavailable, bad
        JSON, empty input) — callers must fall back to
        models.prompt_router.PromptRouter's keyword/_coin_class path, never
        leave the user without a response.
        """
        text = (user_text or "").strip()
        if not self.ok or not text:
            return None
        try:
            from qwen_vl_utils import process_vision_info
            msgs = [
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": f'User request: "{text}"'},
            ]
            chat_text = self._proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inp = self._proc(text=[chat_text], images=imgs, videos=vids,
                             padding=True, return_tensors="pt").to(self._dev)
            with self._torch.no_grad():
                out = self._model.generate(**inp, max_new_tokens=150, do_sample=False)
            ans = self._proc.batch_decode(
                [o[len(i):] for i, o in zip(inp.input_ids, out)],
                skip_special_tokens=True)[0].strip()
            return self._parse_intent_json(ans)
        except Exception as e:
            log.debug(f"VLM parse_capture_intent failed: {e}")
            return None

    @staticmethod
    def _parse_intent_json(raw: str) -> "dict | None":
        """Robustly pull the JSON object out of the model's reply — it
        sometimes wraps output in ```json fences or adds a stray sentence
        despite being told not to."""
        candidate = raw.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
        if fence:
            candidate = fence.group(1)
        else:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = candidate[start:end + 1]
        try:
            data = json.loads(candidate)
        except Exception as e:
            log.debug(f"intent JSON parse failed: {e} — raw: {raw[:200]!r}")
            return None
        if not isinstance(data, dict):
            return None
        # "stop" responses legitimately have no clip_prompt (there's no new
        # objective — see api/server.py's _dispatch_custom_prompt, which
        # clears the current one and returns without touching CLIP at all).
        # Every other capture_mode needs one to be actionable.
        if data.get("capture_mode") != "stop" and not data.get("clip_prompt"):
            return None
        # Sanitize detectors to the known set — never let a hallucinated
        # detector name reach the pipeline.
        data["detectors"] = [d for d in (data.get("detectors") or [])
                              if d in _KNOWN_DETECTORS]
        data.setdefault("capture_mode", "replace")
        data.setdefault("needs_clarification", False)
        data.setdefault("watching_for", data.get("clip_prompt", "")[:40])
        data.setdefault("guide", "I'm watching for it.")
        data.setdefault("reply", "Got it — I'll watch for that.")
        return data

    def derive_event_moments(self, event_name: str) -> "dict | None":
        """General Mode's planning step: turn just an event/occasion NAME
        into a list of the moments worth capturing — no pre-built taxonomy
        (unlike birthday) and no per-moment prompting from the user (unlike
        Custom Mode's one-objective-at-a-time flow). One LLM call, drawing
        on the model's general knowledge of the occasion.

        TEXT-ONLY, same loaded model, same blocking-call contract as
        parse_capture_intent() — callers MUST dispatch this off the asyncio
        event loop (see api/server.py's _dispatch_custom_prompt, which
        General Mode reuses). Returns None on any failure/empty input; the
        caller falls back to a generic candid/group-moment baseline rather
        than leaving the user with nothing.
        """
        text = (event_name or "").strip()
        if not self.ok or not text:
            return None
        try:
            from qwen_vl_utils import process_vision_info
            msgs = [
                {"role": "system", "content": _EVENT_MOMENTS_SYSTEM_PROMPT},
                {"role": "user", "content": f'Event: "{text}"'},
            ]
            chat_text = self._proc.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inp = self._proc(text=[chat_text], images=imgs, videos=vids,
                             padding=True, return_tensors="pt").to(self._dev)
            with self._torch.no_grad():
                # A whole moment LIST is a bigger generation than one intent
                # object (~150 tokens), but this is a LIVE wait, not an
                # album-time background call — generation time scales with
                # this budget almost linearly on this hardware (measured:
                # 150 tokens ≈5s, 500 tokens ≈20s warm), so the budget IS
                # the latency knob. Sized for the system prompt's tightened
                # 4-6 moments × 10-15 word clip_prompt shape (~230-260
                # tokens of real content) with headroom, not for the old
                # 5-8 × open-ended shape this used to allow.
                out = self._model.generate(**inp, max_new_tokens=300, do_sample=False)
            ans = self._proc.batch_decode(
                [o[len(i):] for i, o in zip(inp.input_ids, out)],
                skip_special_tokens=True)[0].strip()
            return self._parse_moments_json(ans)
        except Exception as e:
            log.debug(f"VLM derive_event_moments failed: {e}")
            return None

    @staticmethod
    def _parse_moments_json(raw: str) -> "dict | None":
        candidate = raw.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
        if fence:
            candidate = fence.group(1)
        else:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = candidate[start:end + 1]
        try:
            data = json.loads(candidate)
        except Exception as e:
            # The generation budget is intentionally tight (this is a live
            # wait, not a background call — see derive_event_moments), which
            # means the model can genuinely run past it mid-object: it
            # asked for 4-6 moments but writes a 5th or 6th anyway, and gets
            # cut off partway through it. The `}` that closes the LAST
            # *complete* moment isn't the outermost object's `}` — it's
            # nested inside "moments": [...], so the naive rfind("}") trim
            # above leaves the array/object unclosed and json.loads fails
            # outright, discarding every moment that DID complete along
            # with the one that didn't. Measured live: a real "Diwali" call
            # produced 5 good, complete moments before hitting the ceiling
            # on the 6th — all 5 were being thrown away for one incomplete
            # one. Recover by walking backward through every '}' in the
            # string, closing the array+object after each, until one
            # position yields valid JSON — that's the last COMPLETE moment.
            repaired = None
            idx = candidate.rfind("}")
            while idx != -1:
                try:
                    repaired = json.loads(candidate[:idx + 1] + "]}")
                    break
                except Exception:
                    idx = candidate.rfind("}", 0, idx)
            if repaired is None:
                log.debug(f"event-moments JSON parse failed (unrecoverable): {e} — "
                          f"raw: {raw[:300]!r}")
                return None
            log.info("event-moments JSON was truncated — recovered the moments "
                     "that completed before the cutoff")
            data = repaired
        if not isinstance(data, dict) or not data.get("understood"):
            return None
        moments = data.get("moments")
        if not isinstance(moments, list) or not moments:
            return None
        clean = []
        for m in moments:
            if not isinstance(m, dict):
                continue
            clip_prompt = str(m.get("clip_prompt") or "").strip()
            if not clip_prompt:
                continue
            label = str(m.get("label") or clip_prompt[:30]).strip()
            detectors = [d for d in (m.get("detectors") or []) if d in _KNOWN_DETECTORS]
            clean.append({"label": label, "clip_prompt": clip_prompt, "detectors": detectors})
        if not clean:
            return None
        data["moments"] = clean[:6]   # cap matches the system prompt's "4 to 6" — a
                                       # runaway list would mean more CLIP prompts
                                       # scored every frame than were actually asked for
        data.setdefault("event_summary", "")
        return data

    def verify_moment(self, image_bgr, description: str) -> "bool | None":
        """Binary check: does this ONE frame genuinely show `description`?
        Same pattern as is_birthday() above — a yes/no question is far more
        reliable from a 3B VLM than trusting a raw CLIP similarity score.

        Exists because General Mode's moment vocabulary is free-form and
        often culturally/compositionally specific ("sister tying rakhi on
        brother's wrist") — exactly the kind of fine-grained relationship
        CLIP is known to score unreliably (it grasps rough scene gist, not
        WHO is doing WHAT to WHOM). Measured live: CLIP fired on a random
        solo photo of someone holding a paper packet at a "sister tying
        rakhi" score of 0.08 — indistinguishable from noise. This is the
        gate that catches that before it's ever saved.

        Called on ONE already-CLIP-triggered candidate frame, not every
        frame — so its cost (image VLM call, same as is_birthday()) is
        bounded to roughly once per moment's cooldown window, not per-frame.

        Returns None on any failure (VLM unavailable, bad frame) — the
        caller must NOT veto on None, only on an explicit False (same
        never-block-on-uncertainty contract as is_birthday()).
        """
        if not self.ok or image_bgr is None or not (description or "").strip():
            return None
        try:
            import cv2
            from PIL import Image
            from qwen_vl_utils import process_vision_info
            h, w = image_bgr.shape[:2]
            m = max(h, w)
            if m > 512:
                s = 512.0 / m
                image_bgr = cv2.resize(image_bgr, (max(1, int(w * s)), max(1, int(h * s))))
            pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": (
                    f"Look carefully at this photo. Does it reasonably show: "
                    f"{description.strip()}? Say YES if this is a genuine, "
                    f"recognizable attempt at that moment, even if the framing, "
                    f"angle or lighting isn't perfect. Say NO only if the photo "
                    f"clearly shows something else entirely, unrelated to that "
                    f"moment. Answer strictly YES or NO.")}]}]
            text = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            inp = self._proc(text=[text], images=imgs, videos=vids,
                             padding=True, return_tensors="pt").to(self._dev)
            with self._torch.no_grad():
                out = self._model.generate(**inp, max_new_tokens=4, do_sample=False)
            ans = self._proc.batch_decode(
                [o[len(i):] for i, o in zip(inp.input_ids, out)],
                skip_special_tokens=True)[0].strip().upper()
            if ans.startswith("YES"):
                return True
            if ans.startswith("NO"):
                return False
            return None
        except Exception as e:
            log.debug(f"VLM verify_moment failed: {e}")
            return None


def parse_custom_intent(user_text: str) -> "dict | None":
    """Module-level convenience wrapper: VLMTagger.get() + .parse_capture_intent()
    in one call. Callers on an asyncio event loop (api/server.py's
    _dispatch_custom_prompt) MUST run this whole thing inside a worker
    executor, not just the parse call — VLMTagger.get() itself can block for
    the ~20s cold model load on an unlucky first call, and that must never
    happen on the event loop thread either."""
    return VLMTagger.get().parse_capture_intent(user_text)


def derive_moments(event_name: str) -> "dict | None":
    """Module-level convenience wrapper for General Mode, mirroring
    parse_custom_intent() above — same executor-dispatch requirement."""
    return VLMTagger.get().derive_event_moments(event_name)


# ── General Mode's non-VLM fallback ────────────────────────────────────────
# derive_event_moments() returns None when the VLM is off, still loading, or
# fails to parse — previously that meant EVERY event (a wedding, a cricket
# match, Diwali, an office offsite) landed on the exact same 3 generic
# moments ("a happy moment" / "a group gathering" / "a candid moment"), which
# throws away everything General Mode is supposed to be good at: tailoring
# to what the user actually said. This is a cheap, no-model keyword match
# against the event name — not as good as the VLM's real-world-knowledge
# read, but far closer to it than the flat generic baseline, and it costs
# nothing (no GPU, no latency) so there's no reason not to have it sit
# between "VLM understood it" and "totally generic".
#
# Each category's list mixes moment KINDS the same way the system prompt
# above asks the VLM to (signature ritual/action, social/group, emotional/
# candid, object/setting) — near-duplicate entries hurt live matching just
# as much here as they do in the VLM path.
_FALLBACK_CATEGORIES: "list[tuple[tuple[str, ...], list[dict]]]" = [
    (("wedding", "bride", "groom", "shaadi", "nikah", "marriage"), [
        {"label": "exchanging vows", "clip_prompt": "a couple exchanging vows or rings at an altar", "detectors": ["face"]},
        {"label": "first dance", "clip_prompt": "a couple dancing together as everyone watches", "detectors": ["motion"]},
        {"label": "toast", "clip_prompt": "guests raising glasses in a toast", "detectors": []},
        {"label": "family portrait", "clip_prompt": "a large family group posed together for a photo", "detectors": ["face"]},
        {"label": "emotional reaction", "clip_prompt": "a guest with tears of joy or an emotional smile", "detectors": ["emotion"]},
    ]),
    (("birthday", "bday"), [
        {"label": "blowing candles", "clip_prompt": "a person blowing out candles on a birthday cake", "detectors": ["lighting"]},
        {"label": "cutting cake", "clip_prompt": "a person cutting a decorated birthday cake", "detectors": []},
        {"label": "opening gifts", "clip_prompt": "a person opening a wrapped birthday gift", "detectors": ["emotion"]},
        {"label": "singing together", "clip_prompt": "a group singing together around a cake", "detectors": ["face", "emotion"]},
        {"label": "group celebration", "clip_prompt": "a group of people cheering and celebrating together", "detectors": ["emotion", "motion"]},
    ]),
    (("graduation", "convocation", "commencement"), [
        {"label": "cap toss", "clip_prompt": "graduates throwing their caps into the air", "detectors": ["motion"]},
        {"label": "receiving diploma", "clip_prompt": "a graduate shaking hands and receiving a diploma on stage", "detectors": []},
        {"label": "family celebration", "clip_prompt": "a graduate posing with family members, smiling", "detectors": ["face", "emotion"]},
        {"label": "gown portrait", "clip_prompt": "a graduate in a cap and gown posing for a photo", "detectors": ["face"]},
    ]),
    (("baby shower", "newborn", "baby"), [
        {"label": "opening gifts", "clip_prompt": "someone opening baby gifts surrounded by guests", "detectors": ["emotion"]},
        {"label": "cake moment", "clip_prompt": "a decorated cake being cut or served", "detectors": []},
        {"label": "guests gathered", "clip_prompt": "a group of guests seated together talking and smiling", "detectors": ["face"]},
        {"label": "joyful reaction", "clip_prompt": "an emotional or joyful expression on someone's face", "detectors": ["emotion"]},
    ]),
    (("retirement", "farewell"), [
        {"label": "giving a speech", "clip_prompt": "someone giving a speech in front of a seated group", "detectors": []},
        {"label": "gift presentation", "clip_prompt": "a person being handed a farewell gift", "detectors": ["emotion"]},
        {"label": "group photo", "clip_prompt": "a group of colleagues posed together smiling", "detectors": ["face"]},
        {"label": "toast", "clip_prompt": "people raising glasses in a toast together", "detectors": []},
    ]),
    (("haldi", "mehndi", "mehendi", "sangeet"), [
        {"label": "applying turmeric/henna", "clip_prompt": "turmeric paste or henna being applied to a person's hands, feet or face", "detectors": ["face"]},
        {"label": "dance performance", "clip_prompt": "people dancing together at a pre-wedding celebration", "detectors": ["motion"]},
        {"label": "playful teasing", "clip_prompt": "family and friends laughing and playfully teasing the bride or groom", "detectors": ["emotion"]},
        {"label": "decor close-up", "clip_prompt": "colorful marigold or floral decor at an Indian pre-wedding event", "detectors": []},
        {"label": "group photo", "clip_prompt": "a group of relatives posed together smiling at a family celebration", "detectors": ["face"]},
    ]),
    (("dussehra", "navratri", "garba", "durga puja", "ganesh chaturthi"), [
        {"label": "idol/ritual moment", "clip_prompt": "a traditional idol, shrine or ritual offering being performed", "detectors": ["lighting"]},
        {"label": "dance/garba", "clip_prompt": "people dancing together in traditional festive attire", "detectors": ["motion"]},
        {"label": "family gathering", "clip_prompt": "an extended family or community gathered together for a festival", "detectors": ["face"]},
        {"label": "festive food", "clip_prompt": "traditional festive food or sweets being served", "detectors": []},
    ]),
    (("diwali", "eid", "hanukkah", "christmas", "lunar new year", "rakhi",
      "raksha bandhan", "holi", "thanksgiving", "festival", "puja"), [
        {"label": "lighting ritual", "clip_prompt": "a person performing a traditional ritual or lighting lamps or candles", "detectors": ["lighting"]},
        {"label": "family gathering", "clip_prompt": "an extended family gathered together indoors", "detectors": ["face"]},
        {"label": "festive food", "clip_prompt": "traditional festive food or sweets being served", "detectors": []},
        {"label": "exchanging gifts", "clip_prompt": "people exchanging gifts or blessings with smiles", "detectors": ["emotion"]},
    ]),
    (("student project", "college project", "school project", "assignment",
      "thesis", "capstone", "hackathon", "science fair", "exhibition"), [
        {"label": "presenting work", "clip_prompt": "a student presenting or explaining a project to others", "detectors": []},
        {"label": "hands-on building", "clip_prompt": "a student actively working on or building a project", "detectors": ["motion"]},
        {"label": "team collaboration", "clip_prompt": "a small group of students working together around a table or screen", "detectors": ["face"]},
        {"label": "final result close-up", "clip_prompt": "a close-up of a finished project, model, poster or prototype", "detectors": []},
        {"label": "proud reaction", "clip_prompt": "a student smiling or reacting with pride after finishing or presenting", "detectors": ["emotion"]},
    ]),
    (("monument", "heritage site", "historical site", "fort", "tomb", "palace",
      "sightseeing", "tourist spot", "landmark", "qutub minar", "red fort",
      "humayun", "taj mahal", "agra fort"), [
        {"label": "landmark establishing shot", "clip_prompt": "a person standing in front of a famous monument or landmark, the structure clearly visible", "detectors": []},
        {"label": "candid exploring", "clip_prompt": "a person or group candidly walking through or exploring a historic site", "detectors": ["motion"]},
        {"label": "group/couple pose", "clip_prompt": "a couple or small group posing together with the monument in the background", "detectors": ["face"]},
        {"label": "architectural detail", "clip_prompt": "a close-up of carvings, arches or architectural detail of the monument", "detectors": []},
        {"label": "golden hour scenic", "clip_prompt": "a scenic wide shot of the monument in warm late-afternoon or golden-hour light", "detectors": ["lighting"]},
    ]),
    (("tinder", "bumble", "hinge", "dating app", "dating profile", "dating photos"), [
        {"label": "genuine smile close-up", "clip_prompt": "a close-up portrait of a person with a genuine, relaxed smile", "detectors": ["emotion", "face"]},
        {"label": "lifestyle candid", "clip_prompt": "a candid shot of a person doing an everyday activity or hobby, not posed", "detectors": ["motion"]},
        {"label": "full-body outfit shot", "clip_prompt": "a full-body shot of a person showing their outfit and style", "detectors": []},
        {"label": "social/group context", "clip_prompt": "a person naturally interacting with friends in a social setting", "detectors": ["face", "emotion"]},
    ]),
    (("cloud kitchen", "food photography", "product photography", "e-commerce",
      "flat lay", "flatlay", "menu shoot", "packaging shot", "product shoot",
      "catalog shoot", "catalogue shoot"), [
        {"label": "hero plating shot", "clip_prompt": "a beautifully plated dish or product shot straight-on as the main hero image", "detectors": []},
        {"label": "overhead flat-lay", "clip_prompt": "a top-down flat-lay shot of food or products arranged on a surface", "detectors": []},
        {"label": "process/action shot", "clip_prompt": "a hands-on action shot of food being prepared, plated or a product being made", "detectors": ["motion"]},
        {"label": "branding/packaging detail", "clip_prompt": "a close-up detail shot of product packaging, branding or logo", "detectors": []},
        {"label": "lifestyle context shot", "clip_prompt": "the food or product shown in a natural, in-use lifestyle setting", "detectors": []},
    ]),
    (("portfolio", "profile shoot", "profile picture", "headshot", "social media shoot",
      "instagram shoot", "content shoot", "personal branding", "photoshoot"), [
        {"label": "posed portrait", "clip_prompt": "a single person posed confidently for a portrait photo", "detectors": ["face"]},
        {"label": "candid expression", "clip_prompt": "a natural, candid expression caught mid-motion or mid-laugh", "detectors": ["emotion"]},
        {"label": "outfit/detail shot", "clip_prompt": "a close-up detail shot of an outfit, accessory or styling", "detectors": []},
        {"label": "environment shot", "clip_prompt": "a person framed within an interesting background or setting", "detectors": []},
    ]),
    (("soccer", "football", "basketball", "cricket", "match", "tournament", "sports"), [
        {"label": "action shot", "clip_prompt": "an athlete in mid-action during play", "detectors": ["motion"]},
        {"label": "team celebration", "clip_prompt": "a team celebrating together after a good play", "detectors": ["emotion", "motion"]},
        {"label": "sideline reaction", "clip_prompt": "spectators reacting with excitement", "detectors": ["emotion"]},
        {"label": "team huddle", "clip_prompt": "a team huddled together before or after the game", "detectors": ["face"]},
    ]),
    (("meeting", "offsite", "conference", "corporate", "office"), [
        {"label": "presentation", "clip_prompt": "a person presenting to a seated audience", "detectors": []},
        {"label": "group discussion", "clip_prompt": "colleagues discussing together around a table", "detectors": ["face"]},
        {"label": "networking", "clip_prompt": "people mingling and talking in small groups", "detectors": ["face"]},
        {"label": "team photo", "clip_prompt": "a team posed together for a group photo", "detectors": ["face"]},
    ]),
]

# Used only when nothing above matches — kept broader (5, not 3) and with
# real detectors on each entry, so even a totally unfamiliar/unmatched event
# name still gets a photographer-ish spread instead of a single flat
# "happy/group/candid" triplet with no detector help.
_FALLBACK_GENERIC = [
    {"label": "arrival & gathering", "clip_prompt": "people arriving and gathering together at the start of an event", "detectors": ["face"]},
    {"label": "main activity", "clip_prompt": "people engaged in the main activity or interaction of the event", "detectors": []},
    {"label": "group photo", "clip_prompt": "a posed group of people smiling together for a photo", "detectors": ["face"]},
    {"label": "candid reaction", "clip_prompt": "a candid unposed moment of genuine laughter or emotion", "detectors": ["emotion"]},
    {"label": "celebration", "clip_prompt": "people celebrating with visible joy, cheering or clapping", "detectors": ["emotion", "motion"]},
]


def fallback_event_moments(event_name: str) -> dict:
    """Non-VLM General Mode planner: keyword-matches `event_name` against
    _FALLBACK_CATEGORIES (first match wins) and falls back further to
    _FALLBACK_GENERIC when nothing matches. Always returns a usable dict
    (never None) — unlike derive_event_moments(), there's no failure mode
    to report here, so callers don't need an is-None branch of their own.
    """
    name = (event_name or "").strip().lower()
    moments = _FALLBACK_GENERIC
    for keywords, cat_moments in _FALLBACK_CATEGORIES:
        if any(kw in name for kw in keywords):
            moments = cat_moments
            break
    return {"event_summary": event_name.strip() if event_name else "",
            "moments": [dict(m) for m in moments]}


def verify_moment(image_bgr, description: str) -> "bool | None":
    """Module-level convenience wrapper, mirroring the others above. Callers
    (api/pipeline.py's _evaluate_general_objective, itself already running
    inside the frame-processing executor) call this directly — it's a plain
    blocking call, same contract as every other method on this class."""
    return VLMTagger.get().verify_moment(image_bgr, description)
