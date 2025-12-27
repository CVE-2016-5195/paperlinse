# Paperlinse Development Guide

Paperlinse is a Personal Document OCR and Management System - a web application
that monitors incoming folders (local or network share) for new documents and
organizes them in configured storage locations.

## Project Structure

```
paperlinse/
├── backend/
│   ├── main.py           # FastAPI app, routes, request/response models
│   ├── database.py       # PostgreSQL database integration
│   ├── ocr.py            # PaddleOCR document processing
│   ├── llm.py            # Ollama LLM integration
│   ├── vision_llm.py     # Vision LLM (Qwen3-VL) integration
│   ├── processor.py      # Document processing pipeline
│   ├── config.py         # Configuration with Fernet encryption
│   ├── shares.py         # SMB/CIFS share mounting utilities
│   ├── requirements.txt  # Python dependencies (FastAPI, Pydantic, etc.)
│   └── requirements-vision.txt  # Optional Vision LLM dependencies
├── frontend/
│   ├── src/
│   │   ├── index.html    # SPA entry point
│   │   ├── app.js        # Vanilla JS application logic
│   │   └── style.css     # Dark theme styling
│   ├── package.json      # Node dependencies (Vite)
│   └── vite.config.js    # Vite build config with API proxy
├── run.sh                # Full startup script
├── stop.sh               # Stop script
└── AGENTS.md             # This file
```

## Build & Run Commands

### Full Application
| Command | Description |
|---------|-------------|
| `./run.sh` | Interactive LLM selection with auto-detection |
| `./run.sh --skip-menu` | Auto-select best model without prompt |
| `./run.sh --model qwen3:0.6b` | Force a specific model |
| `./run.sh --timeout 10` | Set model selection timeout (default: 30s) |
| `./run.sh --no-gpu` | Disable GPU acceleration (CPU only) |
| `./stop.sh` | Stop Paperlinse server |
| `./stop.sh --intel-gpu` | Stop server and IPEX-LLM Ollama |

### Auto-Detection Features
The `run.sh` script automatically:
- **Detects Intel GPU** - Enables IPEX-LLM Ollama if Intel iGPU is present
- **Interactive Model Selection** - Shows available models with recommendations
- **Auto-timeout** - Selects recommended model after 30 seconds if no input

### Available LLM Models (all multilingual, support German)
| Model | Memory | Languages | Description |
|-------|--------|-----------|-------------|
| `granite4:350m` | ~512MB | 12 | IBM, smallest, good JSON output |
| `qwen2.5:0.5b` | ~600MB | 29+ | Small multilingual, good JSON output |
| `qwen3:0.6b` | ~768MB | 100+ | Good balance for low-power devices |
| `gemma3:1b` | ~1GB | 140+ | Google's multilingual model |
| `qwen2.5:1.5b` | ~1.5GB | 29+ | **Recommended**, excellent instruction following |
| `qwen2.5:3b` | ~2.5GB | 29+ | Best quality under 3GB |
| `qwen3:4b` | ~3GB | 100+ | Best quality for 8GB systems |

### Backend Only
| Command | Description |
|---------|-------------|
| `cd backend && python main.py` | Run backend server on port 8000 |
| `pip install -r backend/requirements.txt` | Install Python dependencies |

### Frontend Only
| Command | Description |
|---------|-------------|
| `cd frontend && npm install` | Install Node dependencies |
| `cd frontend && npm run dev` | Dev server with hot reload (proxies `/api` to :8000) |
| `cd frontend && npm run build` | Production build to `dist/` |
| `cd frontend && npm run preview` | Preview production build |

### Deploy Frontend to Backend
```bash
cd frontend && npm run build && cp -r dist ../backend/frontend/
```

## Database Setup

Paperlinse requires PostgreSQL. Install and configure it:

```bash
# Install PostgreSQL
apt install -y postgresql postgresql-contrib

# Start and enable
systemctl start postgresql
systemctl enable postgresql

# Create user and database
sudo -u postgres psql -c "CREATE USER paperlinse WITH PASSWORD 'paperlinse';"
sudo -u postgres psql -c "CREATE DATABASE paperlinse OWNER paperlinse;"
```

The backend connects using:
```
DATABASE_URL=postgresql://paperlinse:paperlinse@localhost:5432/paperlinse
```

## Testing

No test framework is currently configured. When adding tests:
- **Backend**: Use `pytest` with `pytest-asyncio` for async endpoint testing
- **Frontend**: Consider Vitest (integrates with Vite)

## Code Style

### Python (Backend)

#### Import Order
Standard library first, then third-party, then local modules. Blank line between groups:
```python
# Standard library
import os
import json
from pathlib import Path
from typing import Optional, Tuple

# Third-party
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet

# Local modules
from config import AppConfig, ShareCredentials
from shares import get_effective_path, is_network_path
```

#### Type Hints
Required on all function parameters and return types:
```python
def get_document_files(directory: Path) -> list[FileInfo]:
    ...

def mount_share(
    share_path: str,
    username: str = "",
    password: str = "",
    domain: str = ""
) -> Tuple[bool, str, Optional[Path]]:
    ...
```

#### Async Functions
All API endpoints must use `async def`:
```python
@app.get("/api/config")
async def get_config() -> dict:
    ...

@app.post("/api/test-connection")
async def test_connection(request: TestConnectionRequest) -> TestConnectionResponse:
    ...
```

#### Pydantic Models
Use Pydantic for all request/response schemas and configuration:
```python
class ShareCredentials(BaseModel):
    """Credentials for network share access."""
    username: str = ""
    password: str = ""
    domain: str = ""

class TestConnectionRequest(BaseModel):
    path: str
    credentials: CredentialsInput
```

#### Error Handling
- Raise `HTTPException` for API errors
- Wrap risky operations in try/except within endpoints
- Return structured error responses:
```python
try:
    success, message, effective_path = get_effective_path(path, ...)
    if not success:
        return TestConnectionResponse(success=False, message=message, writable=False)
except PermissionError:
    return TestConnectionResponse(
        success=False,
        message=f"Permission denied accessing: {path}",
        writable=False
    )
except Exception as e:
    return TestConnectionResponse(
        success=False,
        message=f"Error accessing path: {str(e)}",
        writable=False
    )
```

#### Security
Encrypt sensitive data (passwords) before persisting to disk:
```python
def encrypt_value(value: str) -> str:
    if not value:
        return ""
    f = Fernet(get_or_create_key())
    return base64.urlsafe_b64encode(f.encrypt(value.encode())).decode()
```

#### Docstrings
Use docstrings on modules, classes, and functions:
```python
"""Paperlinse - Personal Document OCR and Management System."""

def get_effective_path(...) -> Tuple[bool, str, Optional[Path]]:
    """
    Get the effective local path for a given path.
    For network shares, mounts them first.
    For local paths, returns them directly.
    
    Returns:
        Tuple of (success, message, effective_path)
    """
```

### JavaScript (Frontend)

#### Module Pattern
Use ES modules with `type="module"`:
```html
<script type="module" src="./app.js"></script>
```

#### API Helpers
Use consistent `apiGet` and `apiPost` helper functions:
```javascript
async function apiGet(endpoint) {
    const response = await fetch(`/api${endpoint}`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

async function apiPost(endpoint, data) {
    const response = await fetch(`/api${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}
```

#### Async/Await with Error Handling
Always wrap API calls in try/catch:
```javascript
async function loadConfig() {
    try {
        const config = await apiGet('/config');
        // Handle response...
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}
```

#### DOM Queries
Use `getElementById` or `querySelector`:
```javascript
document.getElementById('incoming-path').value = config.incoming_share.path || '';
document.querySelector(`.nav-item[data-page="${page}"]`).classList.add('active');
```

#### Window Exposure
Expose functions to `window` if used in HTML onclick handlers:
```javascript
window.navigateTo = navigateTo;
window.testConnection = testConnection;
```

#### Event Listeners
Use `addEventListener` for DOM events:
```javascript
document.getElementById('settings-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    // Handle submit...
});

document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
});
```

### CSS Style

- Use CSS custom properties (variables) for theming
- Dark theme by default
- BEM-like class naming: `.status-card`, `.status-icon`, `.status-info`
- Mobile-responsive with `@media` queries

## Dependencies

### Backend (Python 3.11+)
- fastapi 0.109.0
- uvicorn[standard] 0.27.0
- pydantic >=2.7.0
- pydantic-settings 2.1.0
- python-multipart 0.0.6
- aiofiles 23.2.1
- cryptography 41.0.7
- pyyaml 6.0.1
- asyncpg (PostgreSQL driver)
- httpx (HTTP client for Ollama)
- paddleocr, paddlepaddle (OCR)

### Frontend
- vite ^5.0.0 (dev dependency)

### System Dependencies
- python3.11-venv
- nodejs / npm
- cifs-utils (for network share mounting)
- postgresql (database)

## Architecture Notes

1. **Backend**: FastAPI REST API serves both API endpoints and static SPA files
2. **Frontend**: Vanilla JavaScript SPA built with Vite
3. **Database**: PostgreSQL for document metadata and processing state
4. **Configuration**: JSON file at `config/settings.json` with Fernet-encrypted secrets
5. **File Storage**: Supports both local paths and SMB/CIFS network shares
6. **Development**: Vite dev server proxies `/api` requests to backend on port 8000

## Intel GPU Support (IPEX-LLM)

Paperlinse supports Intel GPU acceleration for LLM inference using IPEX-LLM's Ollama 
portable package. This is ideal for low-power Intel devices like the N150 processor.

### Supported Hardware
- Intel Core Ultra (Meteor Lake) with Arc iGPU
- Intel Core 11th-14th Gen with Iris Xe iGPU
- Intel Arc A-series discrete GPUs

### How It Works
When using `--intel-gpu` flag:
1. Installs Intel GPU compute runtime (OpenCL + Level Zero) if not present
2. Downloads IPEX-LLM Ollama portable to `.ipex-ollama/` directory
3. Runs Ollama with Intel GPU acceleration
4. Uses `qwen3:0.6b` as default model (optimized for Intel, ~600M parameters)

### Auto-installed Dependencies
When `--intel-gpu` is used, the following packages are automatically installed:
- `intel-opencl-icd` - Intel OpenCL ICD
- `intel-level-zero-gpu` - Level Zero GPU driver
- `level-zero` - Level Zero loader
- `ocl-icd-libopencl1` - OpenCL ICD loader

### Environment Variables
The following are set automatically when using Intel GPU mode:
- `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` - Performance optimization
- `OLLAMA_NUM_GPU=999` - Use all available GPU layers
- `OLLAMA_HOST=0.0.0.0:11434` - Listen on all interfaces

### Recommended Models for Intel GPU (all multilingual)
| Model | Size | Languages | Use Case |
|-------|------|-----------|----------|
| `granite4:350m` | ~512MB | 12 | IBM, smallest, good JSON output |
| `qwen2.5:0.5b` | ~600MB | 29+ | Small multilingual, low memory |
| `qwen3:0.6b` | ~768MB | 100+ | Default, fast, good for low-power |
| `gemma3:1b` | ~1GB | 140+ | Google's multilingual model |
| `qwen2.5:1.5b` | ~1.5GB | 29+ | **Best balance**, excellent JSON |
| `qwen2.5:3b` | ~2.5GB | 29+ | Best quality under 3GB |
| `qwen3:4b` | ~3GB | 100+ | Best quality for 8GB systems |

### Logs
IPEX-LLM Ollama logs are written to `logs/ipex-ollama.log`

## Vision LLM Support (Experimental)

Paperlinse supports an experimental Vision LLM mode using OpenVINO and Qwen3-VL. 
This allows direct image-to-metadata extraction, bypassing the OCR + text LLM pipeline.

### How It Works
Instead of:
1. Image → PaddleOCR → Text → Ollama LLM → Metadata

Vision LLM mode uses:
1. Image(s) → Qwen3-VL (OpenVINO) → Metadata

This can provide better results for:
- Documents with complex layouts
- Multi-page documents (all pages sent as images)
- Documents where OCR struggles (handwriting, poor quality scans)

### Installation

1. Install Vision LLM dependencies:
```bash
pip install -r backend/requirements-vision.txt
```

2. Download a pre-converted model:
```bash
# Recommended: Qwen3-VL 2B INT4 (optimized for Intel)
huggingface-cli download helenai/Qwen3-VL-2B-Instruct-int4 --local-dir models/Qwen3-VL-2B-Instruct-int4
```

3. Enable in the web UI:
   - Go to Settings → Vision LLM
   - Check "Enable Vision LLM"
   - Set the model path to your downloaded model
   - Select device (CPU, GPU, or AUTO)

### Available Vision Models
| Model | Size | Description |
|-------|------|-------------|
| `helenai/Qwen3-VL-2B-Instruct-int4` | ~2GB | INT4 quantized, optimized for Intel |

### Requirements
- OpenVINO 2025.3+
- transformers 4.57.1
- Custom optimum-intel branches with Qwen3-VL support (installed from requirements-vision.txt)
- ~4GB RAM minimum (2B model)

### Fallback Behavior
When Vision LLM is enabled:
- If the vision model is available, it will be used for metadata extraction
- OCR still runs (for full-text search and text storage)
- If vision extraction fails, it automatically falls back to text-based extraction

### Device Selection
- **CPU**: Works on any system, slower but compatible
- **GPU**: Requires Intel GPU with OpenVINO support (recommended for Intel devices)
- **AUTO**: Let OpenVINO select the best available device

### Troubleshooting

**Model not loading:**
- Ensure the model path is correct (should contain `openvino_model.xml`)
- Check you have sufficient RAM (~4GB for 2B model)
- Try CPU device first, then GPU

**Slow inference:**
- First run compiles the model (may take 1-2 minutes)
- Subsequent runs should be faster
- Use GPU device on Intel systems for better performance

**Import errors:**
- Ensure you installed from the custom optimum-intel branch
- Run: `pip install -r backend/requirements-vision.txt`

