# SnapAI — Sprint Ownership Map

For every remaining task in the launch roadmap, who's on point.

Legend:
- 🤖 **ME** — I do alone. You say "go" and I deliver.
- 🤝 **BOTH** — I write the code; you provide a key/account/asset I can't generate.
- 👤 **YOU** — Only you can do this. No developer substitution exists.

---

## SPRINT 1 — "Shippable to friends" (weeks 1–2)

| Task | Owner | What blocks it | Effort |
|---|---|---|---|
| Enforce JWT on `/sessions/*` routes | 🤖 ME | nothing | 1 day |
| Add `/auth/register` + `/auth/login` HTTP endpoints | 🤖 ME | nothing | 1 day |
| Add `owner_id` column to sessions, enforce isolation | 🤖 ME | nothing | 1 day |
| Email verification flow (sign-up → verify link → activate) | 🤝 BOTH | YOU: Mailgun/SendGrid free-tier API key | 1 day |
| Forgot-password flow | 🤝 BOTH | same email service | half day |
| Server-side eval harness — runs full pipeline on a video, scores precision/recall vs your timestamp annotations | 🤝 BOTH | YOU: 1 short event video + a CSV of "keeper moment" timestamps | 1 day |
| Sprint-1 unit tests + smoke tests | 🤖 ME | nothing | half day |

**You provide before I start:** Mailgun OR SendGrid API key (signup is free), one short event video with keeper timestamps.

---

## SPRINT 2 — "Deploy to AWS" (weeks 3–4)

| Task | Owner | What blocks it | Effort |
|---|---|---|---|
| Terraform module: VPC + EC2 (single instance) + RDS Postgres + S3 bucket + IAM | 🤝 BOTH | YOU: AWS account + IAM credentials | 2 days |
| `pip install psycopg2`; rewrite `utils/persistence.py` to support both SQLite and Postgres via env var | 🤖 ME | nothing | 2 days |
| Replace local `captures/` filesystem with S3 boto3 uploads | 🤝 BOTH | YOU: AWS account active | 1 day |
| Signed CloudFront URLs for album sharing (expire after 7 days) | 🤝 BOTH | AWS | 1 day |
| Health check + auto-restart via systemd unit | 🤖 ME | nothing | half day |
| CloudWatch log shipping | 🤝 BOTH | AWS | half day |
| Domain DNS pointing to load balancer | 👤 YOU | – | 30 min |
| HTTPS via AWS Certificate Manager | 👤 YOU + 🤖 ME | YOU: domain | 1 hour |
| Reduce Docker image size (multi-stage build) | 🤖 ME | nothing | 1 day |

**You provide before I start:** AWS account with billing alarm at $100/mo, IAM user with admin access (we'll lock down later), domain registered + nameservers under your control.

**You decide:** EU vs US region for first deployment (matters for GDPR).

---

## SPRINT 3 — "iOS mobile app" (weeks 5–7)

| Task | Owner | What blocks it | Effort |
|---|---|---|---|
| React Native scaffold + navigation | 🤖 ME | nothing | 2 days |
| Camera screen with native iOS camera APIs | 🤖 ME | nothing | 3 days |
| WebSocket client to existing server | 🤖 ME | nothing | 1 day |
| Album viewer + share sheet integration | 🤖 ME | nothing | 2 days |
| Login / register screens | 🤖 ME | nothing | 2 days |
| Push notifications (capture alerts) | 🤝 BOTH | YOU: Apple Developer Program + APNs cert | 2 days |
| TestFlight build | 🤝 BOTH | YOU: provisioning profile + bundle ID | 1 day |
| App icons / splash screen artwork | 👤 YOU | designer or AI image gen | 1 day (yours) |
| App Store screenshots (5 sizes per iOS device) | 👤 YOU | – | 2–3 hours |
| App Store description, keywords, privacy disclosures | 👤 YOU | – | 1–2 hours |
| Privacy policy URL (must be public before submission) | 👤 YOU | template + lawyer review | varies |
| Beta-test recruiting (5+ users) | 👤 YOU | – | ongoing |
| QA — actually run on 3 real events | 👤 YOU | – | varies |

**You provide before I start:** Apple Developer Program enrolment ($99/year, 24–48 hr verification).

**Hard truth:** App Store review takes 1–7 days and they reject apps for surprising reasons (camera privacy disclosure language, vague descriptions, weak onboarding). Budget 2 cycles.

---

## SPRINT 4 — "Billing" (weeks 8–9)

| Task | Owner | What blocks it | Effort |
|---|---|---|---|
| Stripe Checkout integration (subscription tier selection → payment → webhook) | 🤝 BOTH | YOU: Stripe account + API keys | 3 days |
| Webhook handler for `customer.subscription.*` events | 🤖 ME | nothing | 1 day |
| Trial-end logic: downgrade after 14 days unless paid | 🤖 ME | nothing | 1 day |
| Settings page (cancel subscription, update payment method, see usage) | 🤖 ME | nothing | 2 days |
| Usage counters in DB (events/month per user) | 🤖 ME | nothing | half day |
| Tier enforcement at runtime (Free = 3 events; Pro = unlimited) | 🤖 ME | nothing | 1 day |
| Email templates: welcome, payment failed, trial ending | 🤖 ME | nothing | 1 day |
| Tax handling (Stripe Tax) | 👤 YOU | configure in Stripe dashboard | 30 min |
| Invoice / receipt format | 👤 YOU | mostly Stripe defaults | 30 min |
| Pricing decision (PDF defaults: Free / $49 / $149 / Custom) | 👤 YOU | – | – |
| Refund policy | 👤 YOU | – | – |
| Business entity for Stripe (LLC / sole prop / etc.) | 👤 YOU | accountant / online incorporation | varies |

**You provide before I start:** Stripe account (verified business), pricing tier values, refund policy text.

---

## SPRINT 5 — "Scale + polish" (weeks 10–11)

| Task | Owner | What blocks it | Effort |
|---|---|---|---|
| Redis pub/sub for cross-instance session state | 🤝 BOTH | YOU: ElastiCache or Upstash account | 2 days |
| Load testing with k6 (simulate 100 concurrent cameras) | 🤖 ME | nothing | 1 day |
| Auto-scaling group config (scale up at 70% CPU) | 🤝 BOTH | AWS | 1 day |
| Admin dashboard (revenue, MRR, active sessions, errors) | 🤖 ME | nothing | 3 days |
| Onboarding wizard (first event walkthrough) | 🤖 ME | nothing | 2 days |
| Sentry integration | 🤝 BOTH | YOU: free Sentry account | half day |
| Status page (status.snappy.example.com) | 🤝 BOTH | YOU: Statuspage / Better Stack account | 1 day |
| Backup automation (RDS snapshots, S3 versioning) | 🤖 ME | nothing | half day |
| Disaster recovery test (restore from snapshot) | 🤝 BOTH | AWS | 1 day |

**You provide before I start:** Sentry, Statuspage (or alternative) free accounts.

---

## SPRINT 6 — "Android + public launch" (weeks 12–14)

| Task | Owner | What blocks it | Effort |
|---|---|---|---|
| Android React Native build (~80% reuse from iOS) | 🤖 ME | nothing | 3 days |
| Android camera permissions + native modules | 🤖 ME | nothing | 1 day |
| Google Play Console listing + APK upload | 🤝 BOTH | YOU: Play Console account ($25 one-time) | 1 day |
| Google Play screenshots, copy, content rating | 👤 YOU | – | 2 hours |
| Marketing site (Next.js, hero + features + pricing) | 🤖 ME | nothing | 3 days |
| Marketing copy + brand voice | 👤 YOU | – | varies |
| Demo video (60s, before/after AI captures) | 👤 YOU | screen + edit | 1 day |
| Marketing site domain pointed at Vercel/Netlify | 👤 YOU | – | 30 min |
| Launch announcement: Product Hunt, HN, Reddit, LinkedIn | 👤 YOU | – | 1 day |
| Affiliate / referral system (the PDF said 20% commission) | 🤖 ME | nothing | 2 days |
| Customer support tooling (Intercom widget) | 🤝 BOTH | YOU: Intercom or Crisp account | half day |
| Studio API for white-label use | 🤖 ME | nothing | 3 days |

**You provide before I start:** Play Console account + signing keys, brand-finalised marketing copy, demo video.

---

## NON-SPRINT TASKS (parallel work you should start NOW)

These don't fit a sprint but block launch. Do them while I work.

### 👤 YOU only — start in week 1

| Task | Effort | Why critical |
|---|---|---|
| Domain registration | 1 hour | Sprint 2 needs it |
| AWS account + billing alarm | 1 hour | Sprint 2 |
| Apple Developer enrolment ($99) | 24–48 hr verify | Sprint 3 — it gates iOS work |
| Google Play Console enrolment ($25) | instant | Sprint 6 |
| Stripe business verification | 1–3 days | Sprint 4 |
| **30 hours of labelled real event video** | 20–40 human-hours | The single biggest accuracy lever. No engineering substitute. |
| Privacy policy + Terms of Service (template + lawyer review) | $200–1500 | Required before App Store submission |
| GDPR review (if EU customers) | $1–3K external | Pre-launch |
| External security audit | $3–5K external | Pre-launch (Tier γ only) |
| 5 reference photographer relationships | weeks of outreach | Tier γ — for case studies + word-of-mouth |
| Brand finalised (logo, colours, voice) | designer fee | Sprint 6 |
| Marketing copy + value props | depends on you | Sprint 6 |
| Demo video script + recording | 1 day | Sprint 6 |
| Discord/Slack community setup | 1 hour | Sprint 6+ |
| Trade show calendar (WPPI Mar, Imaging USA Jan) | research | Phase 5 |

---

## What I CAN do that's not in any sprint

If you want me to do something I haven't listed, here's my menu:

### Available on demand from me alone (🤖)

- Any code in the existing stack (refactor, optimise, new endpoints, new ML models)
- Test coverage expansion (currently ~30 tests; can grow to 100+)
- Eval harnesses against any data you provide
- LoRA fine-tune notebook for Colab (you upload + run)
- More ML models (FaceXFormer, EmoNet, anything pip-installable)
- Schedule/Markov next-moment predictor (the cheap one we discussed)
- Documentation, ARCHITECTURE diagrams, ADRs (architecture decision records)
- Bug fixes for anything you screenshot or log
- Performance profiling + optimisation
- Migration scripts (SQLite → Postgres → anything else)
- Internationalisation (i18n) for the web frontend
- Accessibility audit + WCAG fixes

### Available with your asset/key (🤝)

- Anything that needs an external service (AWS, Stripe, Sentry, Mailgun, Apple, Google, Cloudflare, OpenAI/Anthropic API for LLM features) — give me the key, I integrate it
- Anything that needs YOUR data (videos, photos, labels, customer feedback)
- Anything that needs YOUR decision (pricing, copy, tier limits, support hours)

### Things I FUNDAMENTALLY can't do (👤)

- Create accounts that need a real human's identity
- Sign legal documents
- Pay for anything
- Conduct interviews / sales calls / customer development
- Take photos / record demo videos myself
- Submit to App Store / Play Store (must come from your developer account)
- Get press coverage
- Hire people
- Make business strategy decisions

---

## "What do I do this week?" — the simplest version

| Week | YOU do | I do (if you say go) |
|---|---|---|
| **Week 0 (now)** | Buy domain. Create AWS account. Apple Dev enrolment ($99). Sign up Mailgun + Sentry free tiers. Record + label one short event video. | Wait for the above; refactor/eval-harness work that doesn't need accounts. |
| **Week 1–2** | Test sprint-1 build, give feedback, recruit 5 beta testers. | Sprint 1 — auth + isolation + eval harness. |
| **Week 3–4** | Watch the deploy land at your domain. Sign up Stripe. | Sprint 2 — AWS + Postgres + S3. |
| **Week 5–7** | App Store screenshots, marketing copy, recruit beta testers. | Sprint 3 — iOS app + TestFlight. |
| **Week 8–9** | Pricing decisions. Refund policy. Tax setup in Stripe. | Sprint 4 — billing. |
| **Week 10–11** | Sign up Sentry/Statuspage. Test admin dashboard. | Sprint 5 — scale + polish. |
| **Week 12–14** | Demo video. Press kit. Product Hunt prep. Trade show prep. | Sprint 6 — Android + launch. |

---

## TL;DR

- **Things ONLY YOU can do:** account creation, money, real-world data collection, business decisions, legal review, marketing/sales.
- **Things ONLY I can do:** all the code, infrastructure-as-code, tests, ML model integration, eval harnesses.
- **Things we BOTH do:** anything that needs both an account/key (yours) and code wiring (mine).

In the 14-week launch plan, **roughly 70% of effort is mine, 30% is yours** — but your 30% includes the most critical bottlenecks (real event data, accounts, decisions, legal). Without them, my 70% has nothing to ship.

**Send me sprint-1 inputs (Mailgun key + one labelled event video + AWS account) and I start the same day.**
