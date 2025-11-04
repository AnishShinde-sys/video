# Huemo Video Editing Platform

> Modernised frontend + Node.js backend prepared for deployment to static hosting (frontend) and server runtimes (backend).

## Project Layout

```
frontend/  # Static site (HTML/CSS/JS)
backend/   # Express API + Gemini/Seedance video pipeline
```

## Local Development (optional)

The project is optimised for Render, but you can still run it locally:

```bash
# Backend
cd backend
npm install
cp env.sample .env
# populate GEMINI_API_KEY, EACHLABS_API_KEY, keyID/keyName/applicationKey,
# and optionally DECART_API_KEY if you still use the Decart proxy helpers
npm run dev

# Frontend
cd ../frontend
npm install
npm run start
# The dev server URL is printed in the terminal. If your backend runs on a different
# origin during development, append ?apiBase=<your-backend-url> once; the app
# remembers the override for subsequent requests.
```

### Render Configuration

- `render.yaml` describes two services:
  - **video** (web service, Node runtime, `rootDir: backend`)
  - **huemo-frontend** (static site, `rootDir: frontend`)
- Build commands: `npm install` for backend, none for frontend.
- Start command: `npm run start` (backend).
- Health check: `/health`.
- Required environment variables (set via Render dashboard or MCP):
  - `GEMINI_API_KEY`
  - `EACHLABS_API_KEY`
  - `keyID`
  - `keyName`
  - `applicationKey`
  - Optional: `DECART_API_KEY` if you continue to use the Decart proxy utilities.

The frontend defaults to `https://video-536c.onrender.com` but honours the `?apiBase=` override and persists it in `localStorage`.

## Git Workflow

```bash
git add .
git commit -m "chore: prepare huemo app for deployment"
git push origin main
```

## License

MIT


