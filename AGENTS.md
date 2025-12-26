# Paperlinse Development Guide

## Build & Run Commands
- **Backend**: `cd backend && python main.py` (runs on port 8000)
- **Frontend dev**: `cd frontend && npm run dev` (proxies /api to backend)
- **Frontend build**: `cd frontend && npm run build && cp -r dist ../backend/frontend/`
- **Full start**: `./run.sh` (creates venv, installs deps, builds frontend, starts server)

## Code Style

### Python (Backend)
- Use Pydantic models for all request/response schemas and configuration
- Async functions for all API endpoints (`async def`)
- Type hints required on function parameters and returns
- Imports: stdlib first, then third-party, then local (blank line between groups)
- Error handling: raise `HTTPException` for API errors, catch exceptions in endpoints
- Encrypt sensitive data (passwords) before persisting to disk

### JavaScript (Frontend)
- Vanilla JS with ES modules (`type="module"`)
- Expose functions to `window` object if used in HTML onclick handlers
- Use `async/await` for API calls, wrap in try/catch
- API helpers: `apiGet(endpoint)`, `apiPost(endpoint, data)` pattern
- DOM queries use `getElementById` or `querySelector`
