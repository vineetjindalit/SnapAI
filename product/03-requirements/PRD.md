# Product Requirements Document — SnapAI v1

**Owner:** Sachin Sharma · **Status:** Draft v1 · **Market:** Delhi NCR

## 1. Problem
Short, informal events have no good capture option: professional packages are too big and slow to book; free alternatives pull a guest away from the event.

## 2. Users
| User | Goal | Pain |
|---|---|---|
| Host | Moments captured without managing it | Couldn't reach photographers, too expensive, awkward to vet |
| Partner | Extra income from short gigs | Inconsistent flow, cost of pro gear |
| Ops (founder, early) | Dispatch reliably | Manual WhatsApp assignment |

## 3. Goals and non-goals
**Goals (v1):** a paid booking leads to a delivered, curated album on WhatsApp within minutes of the session ending; partner needs no setup.
**Non-goals (v1):** weddings and long formal events; video highlight reels from raw footage; live DSLR tethering; on-device-only AI.

## 4. User stories and acceptance criteria
1. *Host books a session.* Choose duration, pay via Razorpay; booking confirms only after payment success.
2. *Partner is assigned.* Receives WhatsApp message with a link scoped to the booking; no shared login.
3. *Capture (Mobile tier).* App streams frames; AI auto-captures moments from the event spec; partner can also tap a manual shutter. Both write to one photo pool.
4. *Capture (Camera/Pro tier).* Partner imports photos from the OS photo picker via the batch endpoint, optionally several times per event.
5. *Album.* One action culls duplicates and blurry or irrelevant shots, ranks by importance, enhances, selects a curated subset. **Target:** 120 photos processed in well under 2–3 minutes (measured: 14.7 s curation).
6. *Delivery.* Album link sent to the customer on WhatsApp; booking marked Completed; payout logged.

## 5. Functional requirements
- **Event spec auto-resolution:** server resolves a capture spec at partner acceptance (fixed category, then semantic cache, then one-off VLM derivation); cached by event type; zero partner action.
- **Moment detection:** 9-signal weighted ensemble with priority tiers so must-not-miss moments bypass cooldowns.
- **Quality gates:** blur/exposure scoring, auto-enhance, perceptual-hash dedup.
- **Album generation:** importance-aware selection; manual and auto shots reconciled by best-of-cluster.
- **Booking identity:** one `booking_id` links payment, partner, session and album.

## 6. Privacy and trust (corrected claim)
Photos **do** reach and are stored on the server on every tier, because AI scoring runs server-side. Do **not** claim "nothing is stored". Honest claims: encrypted in transit; isolated per booking; access-controlled; raw/unselected photos auto-deleted N days after delivery *(to build)*; no training on customer photos without consent; no third-party sharing. Event photos capture guests' faces, which is personal data under India's DPDP Act (2023): post an in-event notice and get legal review.

## 7. Success metrics
| Metric | Target for pilot |
|---|---|
| Host "would book again" | ≥ 70% of pilot hosts |
| Host considered a pro instead, and still chose SnapAI | Track; no target until baseline |
| Median time session-end to album delivered | ≤ 10 min |
| Album satisfaction (1–5) | ≥ 4.0 |
| Partner no-show rate | < 5% |
| Booking conversion on site | Baseline, then A/B |

## 8. Risks
See `06-roadmap/roadmap-and-risks.md`. Top risk: semi-pro acceptance at junior-pro price.

## 9. Open questions
Commission structure and price band; refund guarantee wording; trust and safety story for a stranger entering a home (verification, ratings); B2B angle; real-customer language.
