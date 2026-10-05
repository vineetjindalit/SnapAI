# Partner Capture Architecture (condensed)

## Flow
```
Customer books → tier chosen → partner accepts
   → capture spec auto-resolved (cached or derived once)
   → partner opens app on the day, taps booking
   → capture (tier-specific) → shared pipeline (enhance, score, dedup)
   → album curated → delivered on WhatsApp
```

## Corrections made along the way
1. **"The model trains itself per event"** is not realistic. There is no labelled data before the event. What works is zero-shot: a local VLM derives 4–6 moments to watch for from the event name, in seconds.
2. **Tethering a DSLR via a laptop** was the wrong goal. Professionals know when to shoot; what they lack is fast culling. Camera/Pro v1 = shoot normally, import afterward through the phone. Live tethering is a v2 premium feature.
3. **"Mobile tier never touches a server"** is false; AI runs server-side. Dropped as a claim. See PRD privacy section.

## Spec cache
`event_spec_cache(cache_key, capture_spec_json, derived_at, hit_count, source)`. Lookup order: fixed keyword category, then semantic similarity to cached entries, then fresh VLM derivation. The first "Rakhi" booking pays a few seconds; every later one is a database read. Refining a popular event type once improves all future bookings.

## Manual plus auto as a safety net
Both paths write independently to one `session.photos` list. The AI keeps scoring frames while the partner shoots manually; album generation treats both identically and keeps the best per moment cluster. Caveat: this protects moments inside the event spec, not unanticipated ones.

## Getting 300–400 photos into the app
| Method | Throughput | 300–400 photos |
|---|---|---|
| Camera Wi-Fi | 2–5 MB/s | 13–27 min |
| Camera USB cable | 15–25 MB/s | 2–3.5 min |
| USB-C SD-card reader | 60–90 MB/s | 30–55 s |

*(Throughput figures are standard published ranges, not benchmarks of SnapAI.)* Recommended: card reader as default, cable as fallback, Wi-Fi last. Better still: import progressively during natural breaks, and optionally a two-pass flow (small JPEGs for culling, full-res only for the winners).

**Import screen:** booking → "Import Photos" → OS photo picker → `POST /sessions/{sid}/photos/batch` → progress ("118/120") → Generate Album.

## Concurrency
Live Mobile sessions are compute-sensitive (CPU fine for 1–3 simultaneous; beyond that a shared GPU tier). Batch Camera/Pro processing is forgiving. Backpressure drops frames past 6 in flight rather than crashing.

## Reels
Photo-based short reels (Ken-Burns plus beat-synced cuts from the curated album) are cheap and could ship alongside v1. Raw-video highlight editing is a separate phase-2 build.

## Needs paid services
Only GPU capacity for many concurrent live sessions, and camera-brand SDKs if live tethering is built. Everything else runs on the open-source stack.
