#!/bin/bash
# Paperlinse stop script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# IPEX-LLM Ollama directory
IPEX_OLLAMA_DIR="$SCRIPT_DIR/.ipex-ollama"

# Default settings
USE_INTEL_GPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --intel-gpu)
            USE_INTEL_GPU=true
            shift
            ;;
        --help)
            echo "Usage: ./stop.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --intel-gpu   Also stop IPEX-LLM Ollama (Intel GPU mode)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Stop IPEX-LLM Ollama (if --intel-gpu flag is used)
# ============================================================================

stop_ipex_ollama() {
    log_info "Stopping IPEX-LLM Ollama..."
    
    # Stop by PID file
    if [ -f "$IPEX_OLLAMA_DIR/ollama.pid" ]; then
        PID=$(cat "$IPEX_OLLAMA_DIR/ollama.pid")
        if kill -0 "$PID" 2>/dev/null; then
            log_info "Stopping IPEX-LLM Ollama (PID: $PID)..."
            kill "$PID" 2>/dev/null || true
            sleep 2
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                log_warn "Force killing IPEX-LLM Ollama..."
                kill -9 "$PID" 2>/dev/null || true
            fi
            rm -f "$IPEX_OLLAMA_DIR/ollama.pid"
            log_info "IPEX-LLM Ollama stopped"
        else
            log_info "IPEX-LLM Ollama process not running (stale PID file)"
            rm -f "$IPEX_OLLAMA_DIR/ollama.pid"
        fi
    fi
    
    # Also kill by process name as fallback
    if pgrep -f "$IPEX_OLLAMA_DIR/ollama" > /dev/null 2>&1; then
        log_info "Killing remaining IPEX-LLM Ollama processes..."
        pkill -f "$IPEX_OLLAMA_DIR/ollama" 2>/dev/null || true
        sleep 1
    fi
    
    # Kill any ollama processes started by start-ollama.sh
    if pgrep -f "start-ollama.sh" > /dev/null 2>&1; then
        pkill -f "start-ollama.sh" 2>/dev/null || true
    fi
}

# ============================================================================
# Main Stop Logic
# ============================================================================

log_info "Stopping Paperlinse..."

# Kill Python backend by PID file first
if [ -f "$SCRIPT_DIR/.server.pid" ]; then
    PID=$(cat "$SCRIPT_DIR/.server.pid")
    if kill -0 "$PID" 2>/dev/null; then
        log_info "Stopping Paperlinse server (PID: $PID)..."
        kill "$PID" 2>/dev/null || true
        sleep 1
        rm -f "$SCRIPT_DIR/.server.pid"
    else
        rm -f "$SCRIPT_DIR/.server.pid"
    fi
fi

# Also kill by process name as fallback
if pgrep -f "python.*main\.py" > /dev/null; then
    log_info "Stopping Paperlinse Python server..."
    pkill -f "python.*main\.py"
    sleep 1
    log_info "Server stopped"
else
    log_info "No Paperlinse server running"
fi

# Kill Vite dev server if running
if pgrep -f "vite" > /dev/null; then
    log_info "Stopping Vite dev server..."
    pkill -f "vite"
    sleep 1
fi

# Stop IPEX-LLM Ollama if Intel GPU mode was used
if [ "$USE_INTEL_GPU" = true ]; then
    stop_ipex_ollama
fi

log_info "Paperlinse stopped"
