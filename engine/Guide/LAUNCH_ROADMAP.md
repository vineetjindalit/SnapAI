# SnapAI — Launch Roadmap (v2.5 → v3.0 Production)

> **Honest status as of today:** v2.5 is a feature-complete laptop
> prototype with a tier-3 model zoo, online learning, voice prompts,
> auto-classifier, multi-threaded pipeline, JWT auth scaffolding, Docker
> images, and CI templates. **It is not yet a production SaaS.**
>
> **Gap between today and "money-in-the-bank":** approximately 8–14 weeks
> of focused work split between engineering (me) and decisions/spend
> (you). This document tells you exactly what's left and who does it.

---

## 1. Where v2.5 stands vs the PDF roadmap

### Phase 1 — Foundation Hardening: ✅ ~95% done
Everything except 50 hand-written unit tests (we have ~30). Multi-thread
pipeline, MediaPipe, iris gaze, SQLite, logging, env-var tuning are all
in. **Status: production-ready.**

### Phase 2 — AI Model Upgrades: ✅ ~85% done
Real CLIP, NIMA, YOLOv8-face, HSEmotion, online learning, auto-classifier,
calibration are all wired. **Open items:**
- Bootstrap dataset finetune (you're running it)
- LoRA fine-tune of CLIP (needs GPU, ~4 hrs on Colab T4 — not done)
- Confidence calibration on real feedback (works, just needs feedback events)
- A/B harness against real event videos (needs your event data)

### Phase 3 — Product & Mobile App: ⚠️ ~35% done
The shippable-product layer.

| PDF item | Done? | Owner | Effort to finish |
|---|---|---|---|
| Native iOS+Android camera app | ❌ | ME | **3–4 weeks** |
| Polished React web dashboard | ⚠️ basic HTML | ME | 1 week |
| JWT auth enforced + register/login | ⚠️ scaffolding | ME | 2–3 days |
| Album delivery UI (drag-to-reorder, share links) | ⚠️ basic | ME | 4 days |
| Push notifications ("great moment captured!") | ❌ | ME | 3 days |
| Photographer settings panel | ⚠️ env vars only | ME | 2 days |
| QA on 3 real events | ❌ | YOU | varies |
| App Store + Play Store submission | ❌ | YOU + ME | 1–2 weeks |

### Phase 4 — Cloud Infrastructure & Scale: ⚠️ ~25% done
The runs-without-your-laptop layer.

| PDF item | Done? | Owner | Effort to finish |
|---|---|---|---|
| Docker (CPU + GPU) | ✅ | done | – |
| docker-compose for local | ✅ | done | – |
| GitHub Actions CI | ✅ template | YOU adds secrets | 1 hour |
| AWS Terraform module | ❌ | ME | **1 week** |
| S3 photo storage + CloudFront CDN | ❌ | ME | 4 days |
| Migrate SQLite → PostgreSQL (RDS) | ❌ | ME | 1 week |
| Redis pub/sub for multi-instance | ❌ | ME | 4 days |
| Stripe billing integration | ❌ | ME | 1 week |
| Multi-tenant isolation enforced | ⚠️ | ME | 3 days |
| Admin dashboard (revenue, sessions) | ❌ | ME | 1 week |
| Security audit + pen-test | ❌ | YOU (vendor) | $3–5K, 2 weeks |
| GDPR compliance + auto-delete | ❌ | ME | 4 days |
| Load testing (100 simultaneous) | ❌ | ME | 3 days |

### Phase 5 — Market Launch (PDF months 13–18): ❌ 0% done
This is product-market work, not engineering. Mostly your domain. See §5
of this doc.

---

## 2. The minimum viable launch — what MUST be done

I'm splitting this into three release tiers so you can launch incrementally
instead of waiting for everything.

### Tier α — Closed beta (you + 5 friends, no payment)
**Purpose:** validate the product on real events. **Cost:** $0–50.
**Time:** 2 weeks of engineering once we start.

| Must have | Status | Owner |
|---|---|---|
| JWT auth enforced on `/sessions/*` | ⚠️ not enforced | ME (2 days) |
| Register / login endpoints | ❌ | ME (1 day) |
| Per-user data isolation (you can't see my photos) | ❌ | ME (2 days) |
| Domain name with HTTPS | ❌ | YOU ($12/year + 1 hour) |
| Deployable single-container hosting | ⚠️ Docker only | YOU + ME (3 days) |
| Bug-fix the video upload + at least 1 real event tested | ⚠️ | YOU (test) + ME |
| Privacy policy + terms (template, not lawyer-reviewed) | ❌ | YOU (template online, 2 hrs) |

**Result:** snappy.example.com, friends can sign up, upload videos or
stream live, get an album. No money changing hands.

### Tier β — Public beta (paid waitlist, 50 users)
**Purpose:** validate willingness-to-pay. **Cost:** $200–500/mo.
**Time:** +4 weeks engineering on top of α.

| Must have | Status | Owner |
|---|---|---|
| Stripe Checkout (one-time signup fee for waitlist) | ❌ | ME (1 week) |
| AWS deployment (EC2 + RDS + S3) | ❌ | ME (1 week) |
| Postgres migration | ❌ | ME (1 week) |
| Reserved IP / load balancer | ❌ | YOU (config) |
| Polished React Native mobile app (iOS first) | ❌ | ME (3 weeks parallel) |
| Mailgun/SendGrid for transactional email | ❌ | YOU (account) + ME (1 day) |
| Sentry / error tracking | ❌ | YOU (account) + ME (half day) |
| Onboarding flow (first event wizard) | ❌ | ME (1 week) |

### Tier γ — General availability (production paid SaaS)
**Purpose:** real revenue. **Cost:** $500–2000/mo at first 100 users.
**Time:** +6 weeks on top of β.

| Must have | Status | Owner |
|---|---|---|
| Native Android app | ❌ | ME (2 weeks parallel) |
| App Store + Play Store listings | ❌ | YOU ($99 + $25) + ME |
| Referral / affiliate system | ❌ | ME (1 week) |
| Admin dashboard | ❌ | ME (1 week) |
| Security audit (external firm) | ❌ | YOU ($3–5K) |
| GDPR compliance audit | ❌ | YOU (or DPO consultant, $1–3K) |
| Load test ≥ 100 concurrent | ❌ | ME (3 days) |
| Real-event SLA / uptime monitoring | ❌ | ME (2 days) |
| Studio API (white-label) | ❌ | ME (2 weeks) |
| Customer support tooling (Intercom etc.) | ❌ | YOU (account) + ME (1 day) |

---

## 3. Things ONLY YOU can do (no developer can)

These block launch and have no engineering substitute. Do them in
parallel with my work to avoid serial delay.

### Accounts to create (1 day, ~$200 setup)
- [ ] **Domain registration** — Namecheap/Cloudflare ($12/yr)
- [ ] **AWS account** with billing alarms set
- [ ] **Apple Developer Program** ($99/yr) — needed for iOS App Store
- [ ] **Google Play Console** ($25 one-time)
- [ ] **Stripe account** + verified business
- [ ] **Mailgun or SendGrid** (free tier OK)
- [ ] **Sentry** (free tier OK)
- [ ] **Cloudflare** (free tier for DNS + DDoS)
- [ ] **GitHub** organisation + private repo
- [ ] **Hugging Face Hub** (free, for hosting fine-tuned model weights)

### Decisions only you can make
- [ ] Pricing tiers — copy the PDF (Free/Pro/Studio/Enterprise)?
- [ ] Brand name + logo — is "SnapAI" final?
- [ ] Target geography for first launch (US? India? both?)
- [ ] Self-hosted vs managed Postgres
- [ ] Storage retention policy (30 / 60 / 90 days?)
- [ ] Free tier limits (the PDF said 3 events/month, 100 captures)
- [ ] Photographer-friendly licence: who owns the captured photos?

### Money to spend (rough)
| Item | Setup | Monthly |
|---|---|---|
| Domain | $12 | – |
| AWS (Tier β: 1× t3.medium + RDS micro + S3) | – | $80–150 |
| AWS (Tier γ: c5.xlarge + RDS small + ALB + CloudFront + S3) | – | $300–600 |
| Apple/Google dev accounts | $124 | – |
| Stripe fees | – | 2.9% + 30¢/txn |
| Sentry | – | $0–26 |
| Mailgun / SendGrid | – | $0–35 |
| External security audit | $3,000 | – |
| External GDPR review | $1,000 | – |
| **Total to launch** | **~$4,500** | **~$200–700** |

### Real event data to collect (the highest-leverage thing)
- [ ] **Record 5–10 hours of real event footage** (your own / friends' weddings, birthdays). Use a phone — quality doesn't matter for training.
- [ ] **Annotate "good moments"** — just timestamps, e.g. "01:34 cake_cutting peak". Use a Google Sheet, no fancy tool needed.
- [ ] Upload the labelled set to me — I'll fold it into the LoRA fine-tune.

This single deliverable lifts moment accuracy from ~70% to ~85%. Do it
even if you don't do anything else from this list.

### Test events (you must be present)
- [ ] **Event 1**: friend's birthday, your phone running SnapAI on a tripod.
- [ ] **Event 2**: small wedding (or proxy). Get permission to record.
- [ ] **Event 3**: corporate / team event.

---

## 4. Things I can build — concrete sequence

Once you tell me to proceed (and have provided real event data + AWS
account), here's the order I recommend:

### Sprint 1 (week 1–2) — "make it shippable to friends"
- Enforce JWT on session routes (currently scaffolding).
- Build register/login + email verification.
- Per-user session isolation (sid is already random, just add owner_id).
- Polish the album-delivery URL (signed S3 URLs that expire in 7 days).
- Eval harness against your event recordings.

### Sprint 2 (week 3–4) — "deploy to AWS"
- Terraform for VPC + EC2 + RDS + S3 + CloudFront.
- Postgres migration (drop-in for the existing SQLite calls).
- S3 photo storage (replace local filesystem).
- Health checks, auto-restart, log shipping to CloudWatch.

### Sprint 3 (week 5–7) — "iOS app"
- React Native mobile app. Bare-bones first: live camera + connect to
  server + download album.
- TestFlight beta with your 5 friends.

### Sprint 4 (week 8–9) — "billing"
- Stripe integration: trial, subscription, upgrade/downgrade, cancel.
- Settings page.
- Email templates (welcome, billing failed, trial ending).

### Sprint 5 (week 10–11) — "scale + polish"
- Redis pub/sub for multi-instance.
- Load test to 100 concurrent cameras.
- Admin dashboard.
- Onboarding wizard.

### Sprint 6 (week 12–14) — "Android + launch"
- Android React Native build (~80% code reuse from iOS).
- Play Store + App Store submission (review takes 2-7 days).
- Public landing page.

**Total**: 14 weeks of focused engineering. Faster if we parallelise
mobile and backend.

---

## 5. Marketing & GTM (Phase 5 — your domain)

Engineering side (me) is mostly done after sprint 6. Phase 5 in the PDF
is product-market work. I can advise but not execute. Big rocks:

### Pre-launch (during sprints 1–6)
- [ ] Marketing site (snappy.ai or similar). I can build with Next.js.
- [ ] Demo video (60s), before/after captures.
- [ ] Photographer waitlist landing page (email capture).
- [ ] Initial blog content for SEO (5–10 articles, "how AI is changing wedding photography" style).
- [ ] Discord/Slack community started.
- [ ] 5 reference photographers using it free for case studies.

### Launch week
- [ ] Product Hunt launch (aim Tuesday morning Pacific).
- [ ] Hacker News Show HN post.
- [ ] Email blast to waitlist.
- [ ] Reddit r/photography (be careful — they hate self-promo, lead with value).
- [ ] LinkedIn post.

### Post-launch (months 1–6)
- [ ] Trade shows: WPPI ($300 booth, Mar), Imaging USA (Jan), local wedding fairs
- [ ] Influencer photographer partnerships (Justin & Mary, Susan Stripling, etc.)
- [ ] Affiliate program (the PDF said 20% commission)
- [ ] YouTube tutorials
- [ ] Pixieset / Pic-Time / ShootProof integrations

---

## 6. Hard truths

1. **You won't hit the PDF's "$24K MRR by month 18" without a real sales motion.**
   The PDF assumed a marketing lead and 1 sales hire by month 12. Without
   that, you'll be at $5–10K MRR — still useful, but plan for it.

2. **The mobile app is the single biggest engineering chunk.**
   3–6 weeks of work I haven't started. If you don't build it,
   photographers won't take you seriously. The browser app works on phones
   but isn't the product photographers want.

3. **The AI accuracy ceiling is 85% without 30+ hours of YOUR labeled data.**
   No code change I make gets us past that. The bottleneck is data,
   not models. (See ACCURACY_HONEST_TRUTH section in CHANGELOG v2.5.)

4. **Cloud costs scale with photo volume, not user count.**
   At 1,000 events/month × ~100 photos × 2 MB = 200 GB S3 = ~$5 storage.
   Bandwidth is the surprise — CloudFront egress at 1 TB/month is $85.
   Build a realistic cost model BEFORE you set pricing.

5. **GDPR + photo consent is non-negotiable in the EU.**
   You need explicit consent from every face in every photo, OR a
   policy stating the photographer has obtained consent on their end.
   Get the lawyer review before you take EU customers.

6. **The PDF's 18-month timeline assumes a team of 4–5 + $200K seed.**
   We're you + me + AI assistance. Realistic timeline for a real
   launch (Tier γ) is 6–9 months even working full-time, OR 12–18
   months part-time.

---

## 7. The 30-second decision tree

```
Where are you trying to get?

├─ "I want to use this on my own events / friends' events"
│   → You're already there. Just `python3 run.py`.
│
├─ "I want a few photographers to try it for free"
│   → Tier α. 2 weeks of my work + ~$50 setup. I do auth + deploy.
│
├─ "I want to charge money and run a real SaaS"
│   → Tier β minimum (4 more weeks) or Tier γ (12+ more weeks).
│   → ~$4500 setup, ~$300/mo running, 14 weeks engineering total.
│
└─ "I want to be at the PDF's month 18 target ($24K MRR, 200 paying)"
    → Hire 1 React Native engineer + 1 sales/marketer + me.
    → 6–12 months serious investment, $50–150K capital.
```

---

## 8. What I need from you to start sprint 1

1. **One real event video** (5+ minutes, with you marking 3+ keeper moments by timestamp). This is the single most important deliverable.
2. **Decision** on pricing tiers (PDF default is fine if you're undecided).
3. **AWS account** created with a billing alarm at $100/mo.
4. **Domain name** decided + purchased.
5. **Tell me which tier you're aiming at** (α / β / γ).

Once those five are in hand, I start sprint 1 the same day. Sprint 1
unblocks "friends can use it"; everything else is a fork from there.

---

## 9. Pre-launch checklist (the one to print and tape to your wall)

```
Engineering (mine)
[ ] JWT auth enforced + register/login            (sprint 1)
[ ] Eval harness on real event data               (sprint 1)
[ ] AWS Terraform deployment                       (sprint 2)
[ ] Postgres migration                             (sprint 2)
[ ] S3 + signed URLs for albums                    (sprint 2)
[ ] iOS React Native app                           (sprint 3)
[ ] Stripe Checkout integration                    (sprint 4)
[ ] Onboarding wizard                              (sprint 5)
[ ] Admin dashboard                                (sprint 5)
[ ] Android build                                  (sprint 6)
[ ] App Store + Play Store submitted               (sprint 6)

Decisions / spend (yours)
[ ] Domain + AWS + Stripe + Apple + Google accts
[ ] Pricing tiers locked
[ ] Privacy policy + ToS published
[ ] 30 hours of labelled event video
[ ] 5 reference photographer relationships
[ ] Marketing site copy + demo video
[ ] External security audit booked ($3-5K)
[ ] GDPR consultant booked (if EU)
[ ] Trade show calendar (WPPI, Imaging USA)
[ ] Discord / Slack community launched

Validation (mixed)
[ ] 3 real events tested end-to-end
[ ] Load test passes 100 concurrent cameras
[ ] First 10 paying customers identified
[ ] Refund / cancellation flow tested
[ ] Disaster recovery tested (RDS snapshot restore)
```

---

## 10. Final answer to "have you completed Phase 3 and Phase 4?"

**No.**

- Phase 3 is ~35% done — JWT scaffolding, basic dashboard, auth tables. The mobile app, polished UX, and submission to stores are the major remaining chunks.
- Phase 4 is ~25% done — Docker, docker-compose, CI templates. The actual cloud deployment, Postgres migration, Stripe, and admin tooling are the major remaining chunks.

**Realistic time to a paid public launch: 12–14 weeks of engineering + parallel sales/marketing work, assuming you provide AWS, App Store accounts, and at least one labelled event video.**

I can complete every engineering checkbox above. The non-engineering
items (pricing decisions, account creation, real event data, marketing
content, security audit) require you. Tell me which sprint to start and
I'll have it done in the timelines listed.
