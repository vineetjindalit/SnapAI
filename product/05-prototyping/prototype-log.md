# Prototype Log

## What exists
| Piece | State |
|---|---|
| Website and booking funnel | Live (static site on Vercel); Razorpay payment-first; Supabase `bookings` / `partners` |
| WhatsApp (Business Cloud API) | Number verified; templates drafted; not yet auto-triggered |
| AI capture engine ("Snappy") | Runs; sessions can be created for birthday and arbitrary event names |
| Partner mobile app | Scaffolded (`engine/mobile`); icon and wordmark added; not production |

## Engine capabilities (verified by reading the code)
- Real-time capture decision: weighted ensemble over 9 signals with 4 priority tiers.
- Host-group personalisation from face persistence and centrality, no enrolment.
- Album curation: importance-aware selection plus perceptual-hash dedup.
- Auth (JWT), Stripe billing and quota hooks, S3-swappable storage, admin endpoint, voice transcription, training and eval pipeline.

## Verified by running
- Server boots; `/health` OK; frontend served; sessions created for birthday and "haldi ceremony" (General Mode).
- Batch import of **120 real photos: 120 imported, 0 failed, 25.9 s** *(measured)*.
- Album generation on those 120: **41 selected, 14.7 s, 9 distinct moment types** *(measured, VLM refine pass disabled)*.

## Bugs found and fixed
1. `transformers` 5.x broke CLIP loading → pinned `<5`.
2. Whisper loaded unconditionally even for photo-only sessions → now loads only if audio exists.
3. VLM tagger cold-loaded inside the first customer's album → pre-warmed at server boot.

## Features added during prototyping
Manual-photo ingestion, DSLR full-res trigger via camera bridge, batch import endpoint, event catalogue extended (haldi/mehndi, Navratri/Puja, student projects, portfolio, monuments, dating profiles, cloud-kitchen/product).

## Not verified
- Real-device sustained load over 15/30/60 min.
- A real DSLR end to end (`gphoto2`).
- General Mode quality on non-birthday events (tuning was birthday-only).
- Album-time estimates at 500–1000 photos (estimates only).
- HSEmotion model download and SQLite persistence were environment issues in testing; recheck locally.

## Known gaps
- VLM refine pass is birthday-only.
- "Every guest appears at least once" is not guaranteed.
- No raw-photo retention/deletion job yet.

## Integration plan (website ↔ engine)
1. Give every booking a `booking_id`-scoped capture link. 2. Partner opens it on the day. 3. On session end, push album and metadata back keyed to `booking_id`. 4. Auto-deliver on WhatsApp. 5. Mark Completed and log partner payout. Build order: Step 1 first (removes the biggest manual-error risk), then 3–4, then 5.
