# SnapAI — Your Technical Steps to Launch

Every step you (the human) need to take, with the exact command,
expected output, and why each matters. Work through these in order.

---

## 0. Understand what's now in the repo

After the latest commit you have a complete production stack
**stubbed in but not turned on** — the system runs in dev mode by
default; you toggle each subsystem via env vars as you bring services
online.

| Subsystem | Module | Activate with | Default if off |
|---|---|---|---|
| Storage | `backend/storage/` | `SNAPPY_STORAGE=s3` + S3 vars | local filesystem |
| Database | `backend/utils/persistence.py` | `SNAPPY_DB_URL=postgresql://…` | SQLite |
| Cache / pub-sub | `backend/cache/` | `SNAPPY_REDIS_URL=…` | in-memory dict |
| Auth | `backend/auth/` | `SNAPPY_AUTH=1` | open / anonymous |
| Email | `backend/mail/` | `SNAPPY_EMAIL_BACKEND=mailgun` + key | console (logs only) |
| Billing | `backend/billing/` | `SNAPPY_BILLING=stripe` + keys | free tier for everyone |
| Errors | `backend/utils/error_tracking.py` | `SNAPPY_SENTRY_DSN=…` | logs only |

---

## 1. Run it locally (right now, no setup)

```bash
cd /Users/vineetjindal/Downloads/snappy/snappy_final
pip install -r requirements.txt          # ~10 min first time
python3 run.py
```

Open <http://localhost:8765>. Verify:

```bash
curl -s localhost:8765/health | python3 -m json.tool
```

Expected: `version: "v2.5"`, `face.backend: "mediapipe+yolo"`,
`models.clip.available: true`, `models.nima.available: true`.

✅ **You're now at the v2.5 prototype baseline.** Everything below
is for taking it to a paid public SaaS.

---

## 2. Buy a domain (~$12, 10 min)

Go to **Cloudflare Registrar** or **Namecheap**. Pick a domain — `.ai`
costs ~$70/yr; `.app` ~$20; `.com` ~$12.

```bash
# Set up Cloudflare DNS (if you registered there)
# Add an A record for "@" pointing at 192.0.2.1 for now (placeholder)
# We'll point it at the real EC2 IP after Terraform applies
```

**Why now:** sprint 2 needs the domain for HTTPS certs. Cloudflare
nameservers can take up to 24 hours to activate.

---

## 3. Create AWS account (~30 min)

1. Sign up at <https://aws.amazon.com/free>. Use an email you control.
2. **Set a billing alarm** at $100/mo:
   ```
   AWS Console → Billing → Budgets → Create Budget
   Type: Cost budget, Amount: 100 USD/mo, Notifications: 80%, 100%
   ```
3. Create an IAM user with admin access (root account stays untouched):
   ```
   IAM → Users → Add User
   Name: snappy-deploy, Permissions: AdministratorAccess
   Access keys: yes (download CSV)
   ```
4. Configure CLI on your laptop:
   ```bash
   pip install awscli
   aws configure
   # Paste the access key, secret, region (us-east-1), output (json)
   aws sts get-caller-identity     # → should show your user ARN
   ```

**Why now:** Terraform needs these credentials to provision the VPC.

---

## 4. Apple Developer enrolment (~$99, 24-48h verify)

1. Go to <https://developer.apple.com/programs/enroll/>.
2. Pay $99/yr. Use a personal Apple ID OR a business D-U-N-S number.
3. **Wait**: verification takes 24-48 hours. **Start this NOW** — it's
   the longest-pole human task and gates all iOS work.
4. Once approved, add your team ID to the React Native project via
   Xcode > Signing & Capabilities.

**Why now:** sprint 3 (iOS app) is dead in the water without this.

---

## 5. Mailgun (or SendGrid) — free tier, 10 min

For transactional email (verification, welcome, capture notifications).

### Mailgun (recommended — cleaner pricing)

1. Sign up at <https://signup.mailgun.com/new/signup>.
2. Add a domain — **a subdomain is fine** (e.g., `mail.your-snappy-host.com`).
3. Add the DNS records Mailgun shows you (SPF, DKIM, MX) to your DNS
   provider. Wait ~10 min for verification.
4. Get your API key: Dashboard → Settings → API Keys.
5. Test it before you commit:
   ```bash
   export SNAPPY_EMAIL_BACKEND=mailgun
   export SNAPPY_EMAIL_MAILGUN_KEY="key-xxx"
   export SNAPPY_EMAIL_MAILGUN_DOMAIN="mail.your-snappy-host.com"
   export SNAPPY_EMAIL_FROM="SnapAI <noreply@mail.your-snappy-host.com>"
   python3 -c "
   import sys; sys.path.insert(0,'backend')
   from mail.sender import EmailMessage, get_email_sender
   ok = get_email_sender().send(EmailMessage(
       to='YOUR_REAL_EMAIL@example.com',
       subject='SnapAI test', text='If you see this, Mailgun works.'))
   print('sent:', ok)
   "
   ```
   Check your inbox. If empty after 30s, check Mailgun → Logs.

### SendGrid alternative

Same flow, set `SNAPPY_EMAIL_BACKEND=sendgrid` and
`SNAPPY_EMAIL_SENDGRID_KEY=…`.

**Why now:** sprint 1 register/verify flow needs this.

---

## 6. Stripe — when ready to charge (~30 min)

**Skip until sprint 4.** When you're ready:

1. Sign up at <https://dashboard.stripe.com/register>. Verify business.
2. Create products in **Test mode** first:
   ```
   Stripe Dashboard → Products → Add product
   Name: "SnapAI Pro"     Price: $49/mo (recurring) → save the price_id (price_xxx)
   Name: "SnapAI Studio"  Price: $149/mo (recurring) → save price_xxx
   ```
3. Get your test API keys: **Developers → API Keys**.
4. Set webhook endpoint:
   ```
   Developers → Webhooks → Add endpoint
   URL: https://your-snappy-host/billing/webhook
   Events: checkout.session.completed,
           customer.subscription.created,
           customer.subscription.updated,
           customer.subscription.deleted
   → Save the webhook signing secret (whsec_xxx)
   ```
5. Test locally:
   ```bash
   export SNAPPY_BILLING=stripe
   export SNAPPY_STRIPE_SECRET="sk_test_..."
   export SNAPPY_STRIPE_WEBHOOK_SECRET="whsec_..."
   export SNAPPY_STRIPE_PRO_PRICE="price_..."
   export SNAPPY_STRIPE_STUDIO_PRICE="price_..."
   python3 run.py
   curl -s localhost:8765/billing/tiers | python3 -m json.tool
   # → Should list all 4 tiers with prices
   ```
6. When you're ready for real money, swap to live keys (`sk_live_…`).

---

## 7. Deploy to AWS via Terraform (~30 min once accounts ready)

```bash
brew install terraform                                        # macOS
cd /Users/vineetjindal/Downloads/snappy/snappy_final/deploy/terraform

# Copy the example tfvars and fill in
cp secrets.auto.tfvars.example secrets.auto.tfvars
$EDITOR secrets.auto.tfvars
# Set: domain, db_password, container_image (push your Docker image to GHCR first)

terraform init
terraform plan      # Read carefully. Should show ~25 resources to create.
terraform apply     # Type 'yes' to confirm. Takes ~8-10 minutes.
```

Outputs at the end:
- `app_public_ip` — the EC2 instance running SnapAI
- `db_endpoint`  — RDS PostgreSQL host (already wired into the EC2 startup)
- `s3_bucket`    — your photos bucket

**Point your domain DNS** at the public IP:
```
Cloudflare DNS → Add Record
Type: A, Name: @, Content: <app_public_ip>, Proxy: DNS only (for now)
```

Wait 5 min, then:

```bash
curl -s http://<app_public_ip>:8765/health | python3 -m json.tool
# → version v2.5, all models available
```

✅ **You now have a public SnapAI server.** Add HTTPS via AWS Certificate
Manager + ALB next sprint.

---

## 8. Push your Docker image to a registry (~5 min)

```bash
# Build
cd /Users/vineetjindal/Downloads/snappy/snappy_final
docker build -f deploy/Dockerfile -t snappy:v2.5 .

# Push to GitHub Container Registry (free for public repos)
echo $GITHUB_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
docker tag snappy:v2.5 ghcr.io/YOUR_ORG/snappy:v2.5
docker push ghcr.io/YOUR_ORG/snappy:v2.5
```

Update `secrets.auto.tfvars` `container_image` then `terraform apply` again.

---

## 9. Sentry (~5 min, free)

1. Sign up at <https://sentry.io/signup>.
2. Create a project → "Python".
3. Copy the DSN (`https://xxx@oxxx.ingest.sentry.io/xxx`).
4. Add to your env:
   ```bash
   export SNAPPY_SENTRY_DSN="https://xxx@..."
   pip install sentry-sdk
   python3 run.py
   ```
5. Trigger an error to verify: hit a bad route. Check Sentry dashboard.

---

## 10. Test the auth flow end-to-end (~5 min)

```bash
# 1. Enable auth + start server
SNAPPY_AUTH=1 python3 run.py

# 2. In another terminal — register
curl -s -X POST localhost:8765/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"hunter2word"}'
# → {"ok": true, "user_id": 1, "email": "test@example.com"}

# 3. Login → get token
TOKEN=$(curl -s -X POST localhost:8765/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"hunter2word"}' | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
echo $TOKEN

# 4. Use the token to create a session
curl -s -X POST localhost:8765/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"event_name":"Auth Test","event_type":"general","prompt":"general"}'
# → {"session_id": "abc123", ...}

# 5. Confirm anonymous access is blocked
curl -s -X POST localhost:8765/sessions \
  -H "Content-Type: application/json" \
  -d '{"event_name":"Should fail"}'
# → {"error": "Missing Bearer token"}
```

---

## 11. Run the load test (~5 min)

```bash
brew install k6
k6 run -e SNAPPY_HOST=http://localhost:8765 -e VUS=20 \
       deploy/k6_loadtest.js
```

Watch the summary. p99 should stay under 500 ms with 20 VUs on a CPU
laptop. 100 VUs needs the production deployment with workers.

---

## 12. Record your first labelled event video (~30 min)

**This is the highest-leverage thing you can do.** It directly raises
moment-detection accuracy.

1. Record a real event with your phone (5–30 min). Wedding, birthday,
   sports — anything.
2. Open the video in any player. Pause and note timestamps for keeper
   moments. Format:
   ```
   00:34  cake_cutting
   01:12  group_photo
   02:48  ring_ceremony
   ```
3. Upload via the web UI's Video tab, or:
   ```bash
   python3 scripts/test_video_upload.py /path/to/your-video.mp4
   ```
4. Compare the captures SnapAI chose against your timestamps. Mismatches
   become training data for the LoRA fine-tune.

---

## 13. Decisions you must make before launching

Don't ship until each of these is decided. Write your answers in
`launch_decisions.md`:

- [ ] **Brand name + domain** (final)
- [ ] **Pricing**: stick with $0/$49/$149 PDF defaults?
- [ ] **Free tier limits**: 3 events × 100 captures = good?
- [ ] **Photo retention**: 30 / 60 / 90 days? (currently 90 in Terraform)
- [ ] **Geographic launch**: US first? India? Both?
- [ ] **Privacy policy**: lawyer-reviewed or template? ($200–1500)
- [ ] **Photo licence**: who owns photos — photographer or SnapAI?
- [ ] **Refund policy**: 7 / 14 / 30 days?
- [ ] **Support hours**: email-only or live chat?

---

## 14. Pre-launch checklist (the 90-day countdown)

```
T-90 days  ✅ Domain, AWS, Apple, Stripe, Mailgun accounts created
T-90 days  ✅ Sprint 1 done (auth + isolation + eval harness)
T-75 days  ✅ Sprint 2 done (AWS deployed, HTTPS, Postgres)
T-60 days  ✅ Sprint 3 done (iOS app on TestFlight, 5 beta users)
T-45 days  ✅ Sprint 4 done (Stripe live, pricing locked)
T-45 days  ✅ Privacy policy + ToS published
T-30 days  ✅ Sprint 5 done (admin dashboard, Sentry, status page)
T-30 days  ✅ Security audit booked + scheduled
T-21 days  ✅ Sprint 6 done (Android in Play review, marketing site)
T-14 days  ✅ Demo video recorded + edited
T-14 days  ✅ Press kit ready (logo, screenshots, copy)
T-7  days  ✅ Product Hunt account warmed (post 5 supportive comments on others)
T-7  days  ✅ Email blast list ready
T-3  days  ✅ Final smoke test on production
T-1  day   ✅ Status page green
T-0        🚀 Product Hunt + HN + LinkedIn + email blast
T+1  day   ✅ Reply to every signup within 4 hours
T+7  days  ✅ First retrospective: did anyone pay? Why or why not?
T+30 days  ✅ First 10 paying customers, or pivot
```

---

## 15. The summary card

What I built today (you can use these immediately):

| File | What it does |
|---|---|
| `backend/storage/` | Local + S3 storage abstraction |
| `backend/mail/` | Mailgun + SendGrid + console email |
| `backend/auth/` | JWT enforcement, register/login/me, password hashing |
| `backend/billing/` | Stripe checkout + webhooks + tier enforcement |
| `backend/cache/` | In-memory + Redis cache |
| `backend/utils/error_tracking.py` | Sentry init |
| `deploy/terraform/main.tf` | Full AWS deployment (VPC + EC2 + RDS + S3 + Redis) |
| `deploy/k6_loadtest.js` | k6 load test script |
| `mobile/` | React Native scaffold |
| `landing/index.html` | Marketing landing page |

What you do this week:
1. Buy domain + AWS account + Apple enrolment **in parallel**.
2. Test the auth flow locally (§10 above).
3. Run the k6 load test (§11).
4. Record one labelled event video (§12).
5. Tell me which sprint to start.

Once §1–4 above are done, I have everything I need to push you the rest
of the way.
