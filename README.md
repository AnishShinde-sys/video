# Huemo Video Editing Platform

> Modernised frontend + Node.js backend prepared for deployment to static hosting (frontend) and server runtimes (backend).

## Project Layout

```
frontend/  # Static site (HTML/CSS/JS)
backend/   # Express API proxying to Decart LucyEdit
```

## Prerequisites

- Node.js 18+
- Decart API key stored in `.env` (see `backend/env.sample`)

## Local Development

```bash
# Backend
cd backend
npm install
cp env.sample .env  # then edit DECART_API_KEY
npm run dev

# Frontend (in another terminal)
cd ../frontend
python3 -m http.server 9000

# Visit http://localhost:9000 (frontend connects to http://localhost:4000 automatically)
```

## Deployment

### Backend

1. Choose a Node host (Railway, Render, Fly.io, etc.).
2. Configure build command `npm install --prefix backend` and start command `npm run start --prefix backend` (or `cd backend && npm run start`).
3. Set environment variables from `backend/env.sample`:
   - `DECART_API_KEY`
   - optional `PORT`
4. Ensure persistent storage paths (`backend/storage` and `backend/uploads`) exist or use ephemeral storage as fits the platform.

### Frontend

1. Host `frontend/` on any static provider (Vercel, Netlify, S3, etc.).
2. If your backend lives on a different origin, append `?apiBase=https://your-backend.example.com` to the site once; the app stores this value in `localStorage` for future requests.
3. For same-origin deployments (e.g. reverse proxy), no extra configuration is required.

## Git Workflow

```bash
git add .
git commit -m "chore: prepare huemo app for deployment"
git push origin main
```

## License

MIT


