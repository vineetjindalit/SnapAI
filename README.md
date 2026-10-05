<p align="center"><img src="brand/wordmark.png" alt="SnapAI" width="360"></p>

# SnapAI

On-demand, AI-assisted event photography for Delhi NCR. Book a capturer in minutes, get an edited album on WhatsApp before the event ends.

## Repository layout

| Folder | What it is |
|---|---|
| `website/` | Production site (snapai.in): booking + Razorpay payment flow, partner registration, legal pages. Static HTML/CSS/JS on Vercel, Supabase backend. |
| `engine/` | AI smart event capturer ("Snappy"): FastAPI backend, CLIP + face/emotion/pose-based moment scoring, event classifiers, training scripts, web + React + mobile front-ends, Modal/Docker deploy configs, tests. See `engine/README.md` and `engine/ARCHITECTURE.md`. |

## Notes
- Secrets, TLS certs, model weights over a few MB, captured photos, albums and large datasets are deliberately not committed (see `.gitignore`).
- The internal admin dashboard is not published here.

Built by Sachin Sharma.
