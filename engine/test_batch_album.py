import base64, glob, json, time, urllib.request

BASE = "http://localhost:8765"

def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE+path, data=data, headers={"Content-Type":"application/json"}, method="POST")
    t0=time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read()), time.time()-t0

files = sorted(glob.glob("captures/*/*.jpg"))[:120]
print(f"Using {len(files)} real photos for batch-import + album test")

status, resp, t = post("/sessions", {"event_name": "Test Mobile Booking", "event_type": "birthday"})
sid = resp["session_id"]
print(f"session created: {sid} ({t:.2f}s)")

images = [base64.b64encode(open(f,"rb").read()).decode() for f in files]
status, resp, t = post(f"/sessions/{sid}/photos/batch", {"images": images})
print(f"BATCH IMPORT: status={status} imported={resp.get('imported')} failed={resp.get('failed')} time={t:.2f}s")

status, resp, t = post(f"/sessions/{sid}/album", {})
print(f"GENERATE ALBUM: status={status} total_captured={resp.get('total_captured')} total_selected={resp.get('total_selected')} time={t:.2f}s")
print(json.dumps(resp.get("stats", {}), indent=2)[:500])
