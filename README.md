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
# populate GEMINI_API_KEY, EACHLABS_API_KEY, keyID/keyName/applicationKey, DECART_API_KEY (if used)
npm run dev

# Frontend
cd ../frontend
npm install
npm run start  # serves on http://127.0.0.1:8080

# Point the frontend at your local backend once:
# http://127.0.0.1:8080?apiBase=http://127.0.0.1:4000
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


