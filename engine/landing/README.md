# Snappy Marketing Landing Page

Static site for **snappy.example.com**. Generates with Next.js — designed
to be the public-facing splash, pricing, signup, and waitlist.

## Bootstrap

```bash
cd landing
npx create-next-app@latest site --ts --tailwind --eslint --app
cp index.html site/app/page.tsx     # paste the marketing copy
cp -r ../public site/public
cd site
npm run dev
```

Deploy via Vercel (free for low traffic):

```bash
npm install -g vercel
vercel
```

## What's here

- `index.html` — single-file marketing landing page (no build needed for
  preview). Drop into a Next.js project to make it a real React tree.
- Hero + features + pricing + waitlist email capture.
- Pricing tier values pulled from PDF (Free / $49 Pro / $149 Studio / Custom).

## Sprint 6 buildout

- Real email capture (Mailgun list, Buttondown, ConvertKit)
- Demo video embed (YouTube or self-hosted)
- Photographer testimonials carousel
- Blog (`/blog/*` MDX routes)
- Contact / sales form
- Privacy policy + ToS
- OG meta tags + Twitter card
- Plausible / GA analytics

The `index.html` provided is intentionally vanilla — no framework lockin
until you decide between Next.js, Astro, or plain Vercel static.
