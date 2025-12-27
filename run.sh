#!/bin/bash
# Paperlinse startup script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

# Default settings
USE_INTEL_GPU=false
AUTO_DETECT=true
IPEX_OLLAMA_DIR="$SCRIPT_DIR/.ipex-ollama"
IPEX_OLLAMA_VERSION="v2.3.0-nightly"

# ============================================================================
# Hardware Detection Functions
# ============================================================================

detect_intel_gpu() {
    # Check if Intel GPU is present (vendor ID 0x8086)
    if [ -d "/dev/dri" ]; then
        for card in /sys/class/drm/card*/device/vendor; do
            if [ -f "$card" ] && grep -q "0x8086" "$card" 2>/dev/null; then
                return 0
            fi
        done
    fi
    return 1
}

get_available_memory_mb() {
    # Get available memory in MB (not total, but actually available)
    free -m | awk '/^Mem:/ {print $7}'
}

get_total_memory_mb() {
    # Get total memory in MB
    free -m | awk '/^Mem:/ {print $2}'
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --intel-gpu)
            USE_INTEL_GPU=true
            AUTO_DETECT=false
            shift
            ;;
        --no-gpu)
            USE_INTEL_GPU=false
            AUTO_DETECT=false
            shift
            ;;
        --help)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --intel-gpu     Force Intel GPU mode (auto-detected by default)"
            echo "  --no-gpu        Disable GPU acceleration"
            echo "  --help          Show this help message"
            echo ""
            echo "LLM Model Selection:"
            echo "  Models are now configured in the Settings page of the web UI."
            echo "  You can select, download, and switch models from the LLM Settings section."
            echo ""
            echo "Auto-detection:"
            echo "  - Intel GPU is automatically detected and enabled if present"
            echo ""
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Auto-Detection
# ============================================================================

log_info "Detecting hardware..."

# Auto-detect Intel GPU if not explicitly set
if [ "$AUTO_DETECT" = true ]; then
    if detect_intel_gpu; then
        USE_INTEL_GPU=true
        log_info "Intel GPU detected - enabling GPU acceleration"
    else
        log_info "No Intel GPU detected - using CPU mode"
    fi
fi

# Get memory info
TOTAL_MEM=$(get_total_memory_mb)
AVAIL_MEM=$(get_available_memory_mb)
log_info "Memory: ${AVAIL_MEM}MB available / ${TOTAL_MEM}MB total"

# ============================================================================
# Intel GPU Compute Runtime Installation
# ============================================================================

install_intel_gpu_deps() {
    log_info "Checking Intel GPU compute runtime..."
    
    # Check if OpenCL is already available
    if ldconfig -p 2>/dev/null | grep -q "libOpenCL.so.1"; then
        log_info "Intel OpenCL runtime already installed"
        return 0
    fi
    
    log_info "Installing Intel GPU compute runtime (OpenCL + Level Zero)..."
    
    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        OS_VERSION=$VERSION_ID
    else
        log_error "Cannot detect OS. Please install Intel GPU drivers manually."
        log_error "See: https://dgpu-docs.intel.com/driver/client/overview.html"
        exit 1
    fi
    
    case $OS in
        ubuntu)
            log_info "Detected Ubuntu $OS_VERSION"
            
            # Determine Ubuntu codename for Intel repo
            case $OS_VERSION in
                22.04) UBUNTU_CODENAME="jammy" ;;
                24.04) UBUNTU_CODENAME="noble" ;;
                *)
                    log_warn "Ubuntu $OS_VERSION may not be fully supported, trying jammy packages"
                    UBUNTU_CODENAME="jammy"
                    ;;
            esac
            
            # Add Intel graphics repository
            log_info "Adding Intel graphics repository..."
            wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
                gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg 2>/dev/null || \
                gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics.gpg
            
            echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${UBUNTU_CODENAME} unified" \
                > /etc/apt/sources.list.d/intel-gpu.list
            
            # Install packages
            apt-get update -qq
            log_info "Installing Intel OpenCL runtime..."
            apt-get install -y intel-opencl-icd ocl-icd-libopencl1 || {
                log_error "Failed to install Intel OpenCL packages"
                exit 1
            }
            log_info "Installing Intel Level Zero runtime..."
            apt-get install -y libze1 libze-intel-gpu1 || {
                log_warn "Level Zero packages not installed - GPU acceleration may be limited"
            }
            ;;
            
        debian)
            log_info "Detected Debian $OS_VERSION"
            
            # Debian uses Ubuntu packages from Intel (jammy works for Debian 12)
            case $OS_VERSION in
                12) UBUNTU_CODENAME="jammy" ;;
                11) UBUNTU_CODENAME="focal" ;;
                *)
                    log_warn "Debian $OS_VERSION may not be fully supported, trying jammy packages"
                    UBUNTU_CODENAME="jammy"
                    ;;
            esac
            
            # Add Intel graphics repository
            log_info "Adding Intel graphics repository..."
            wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
                gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg 2>/dev/null || \
                gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics.gpg
            
            echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${UBUNTU_CODENAME} unified" \
                > /etc/apt/sources.list.d/intel-gpu.list
            
            # Install packages
            apt-get update -qq
            log_info "Installing Intel OpenCL runtime..."
            apt-get install -y intel-opencl-icd ocl-icd-libopencl1 || {
                log_error "Failed to install Intel OpenCL packages"
                exit 1
            }
            log_info "Installing Intel Level Zero runtime..."
            apt-get install -y libze1 libze-intel-gpu1 || {
                log_warn "Level Zero packages not installed - GPU acceleration may be limited"
            }
            ;;
            
        fedora|centos|rhel)
            log_info "Detected $OS $OS_VERSION"
            
            # Add Intel repository for Fedora/RHEL
            dnf install -y 'dnf-command(config-manager)' 2>/dev/null || yum install -y yum-utils
            dnf config-manager --add-repo https://repositories.intel.com/gpu/rhel/9/lts/2350/unified/intel-gpu-9.repo 2>/dev/null || \
                yum-config-manager --add-repo https://repositories.intel.com/gpu/rhel/9/lts/2350/unified/intel-gpu-9.repo
            
            dnf install -y intel-opencl level-zero intel-level-zero-gpu || {
                log_error "Failed to install Intel GPU packages"
                exit 1
            }
            ;;
            
        arch|manjaro)
            log_info "Detected $OS"
            pacman -Sy --noconfirm intel-compute-runtime level-zero-loader ocl-icd || {
                log_error "Failed to install Intel GPU packages"
                exit 1
            }
            ;;
            
        *)
            log_error "Unsupported OS for automatic Intel GPU driver installation: $OS"
            log_error "Please install Intel GPU compute runtime manually:"
            log_error "  - intel-opencl-icd (OpenCL ICD)"
            log_error "  - intel-level-zero-gpu (Level Zero GPU driver)"
            log_error "  - level-zero (Level Zero loader)"
            log_error "See: https://dgpu-docs.intel.com/driver/client/overview.html"
            exit 1
            ;;
    esac
    
    # Update library cache
    ldconfig
    
    # Verify installation
    if ldconfig -p 2>/dev/null | grep -q "libOpenCL.so.1"; then
        log_info "Intel GPU compute runtime installed successfully"
    else
        log_error "Intel GPU runtime installation may have failed - libOpenCL.so.1 not found"
        log_error "Please check the installation manually"
        exit 1
    fi
    
    # Check for GPU device access
    if [ -d "/dev/dri" ]; then
        if [ -r "/dev/dri/renderD128" ]; then
            log_info "Intel GPU device accessible at /dev/dri/renderD128"
        else
            log_warn "GPU device exists but may not be accessible"
            log_warn "You may need to add your user to the 'render' or 'video' group"
        fi
    else
        log_warn "No /dev/dri found - Intel GPU may not be available"
    fi
}

# ============================================================================
# IPEX-LLM Ollama Installation (Intel GPU Support)
# ============================================================================

install_ipex_ollama() {
    log_info "Setting up IPEX-LLM Ollama for Intel GPU..."
    
    # Create directory for IPEX-LLM Ollama
    mkdir -p "$IPEX_OLLAMA_DIR"
    
    # Check if already installed
    if [ -f "$IPEX_OLLAMA_DIR/ollama" ]; then
        log_info "IPEX-LLM Ollama already installed"
        return 0
    fi
    
    # Detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64|amd64)
            # x86_64 is supported
            ;;
        *)
            log_error "Unsupported architecture for IPEX-LLM: $ARCH"
            log_error "IPEX-LLM Ollama only supports x86_64 architecture"
            exit 1
            ;;
    esac
    
    # Get the latest ollama-ipex-llm download URL from GitHub API
    log_info "Fetching latest IPEX-LLM Ollama release info..."
    DOWNLOAD_URL=$(curl -s "https://api.github.com/repos/ipex-llm/ipex-llm/releases/tags/${IPEX_OLLAMA_VERSION}" | \
        grep -o '"browser_download_url": "[^"]*ollama-ipex-llm[^"]*ubuntu\.tgz"' | \
        tail -1 | \
        cut -d'"' -f4)
    
    if [ -z "$DOWNLOAD_URL" ]; then
        log_error "Failed to find IPEX-LLM Ollama download URL"
        log_error "Please download manually from: https://github.com/ipex-llm/ipex-llm/releases"
        exit 1
    fi
    
    log_info "Downloading IPEX-LLM Ollama from: $DOWNLOAD_URL"
    log_info "This may take a few minutes (~500MB)..."
    
    # Download with progress
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$IPEX_OLLAMA_DIR/ipex-ollama.tgz" "$DOWNLOAD_URL" || {
            log_error "Failed to download IPEX-LLM Ollama"
            log_error "Please check your internet connection and try again"
            log_error "Or download manually from: https://github.com/ipex-llm/ipex-llm/releases"
            exit 1
        }
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$IPEX_OLLAMA_DIR/ipex-ollama.tgz" "$DOWNLOAD_URL" || {
            log_error "Failed to download IPEX-LLM Ollama"
            exit 1
        }
    else
        log_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    
    # Verify download succeeded
    if [ ! -f "$IPEX_OLLAMA_DIR/ipex-ollama.tgz" ] || [ ! -s "$IPEX_OLLAMA_DIR/ipex-ollama.tgz" ]; then
        log_error "Download failed or file is empty"
        rm -f "$IPEX_OLLAMA_DIR/ipex-ollama.tgz"
        exit 1
    fi
    
    # Extract
    log_info "Extracting IPEX-LLM Ollama..."
    tar -xzf "$IPEX_OLLAMA_DIR/ipex-ollama.tgz" -C "$IPEX_OLLAMA_DIR" --strip-components=1 || {
        log_error "Failed to extract IPEX-LLM Ollama"
        rm -f "$IPEX_OLLAMA_DIR/ipex-ollama.tgz"
        exit 1
    }
    rm "$IPEX_OLLAMA_DIR/ipex-ollama.tgz"
    
    # Make executable
    chmod +x "$IPEX_OLLAMA_DIR/ollama" 2>/dev/null || true
    chmod +x "$IPEX_OLLAMA_DIR/start-ollama.sh" 2>/dev/null || true
    
    log_info "IPEX-LLM Ollama installed successfully"
}

start_ipex_ollama() {
    log_info "Starting IPEX-LLM Ollama with Intel GPU acceleration..."
    
    # Check if already running
    if pgrep -f "ipex-ollama.*ollama" > /dev/null || pgrep -f "$IPEX_OLLAMA_DIR/ollama" > /dev/null; then
        log_info "IPEX-LLM Ollama is already running"
        return 0
    fi
    
    # Set Intel GPU environment variables
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
    export OLLAMA_NUM_GPU=999  # Use all available GPU layers
    export OLLAMA_HOST=0.0.0.0:11434
    export OLLAMA_MODELS="$IPEX_OLLAMA_DIR/models"
    
    # Add IPEX-LLM libraries to LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$IPEX_OLLAMA_DIR:${LD_LIBRARY_PATH:-}"
    
    # Create models directory
    mkdir -p "$OLLAMA_MODELS"
    
    # Check for Intel GPU
    if [ -d "/dev/dri" ]; then
        log_info "Intel GPU device found at /dev/dri"
    else
        log_warn "No /dev/dri found - Intel GPU may not be available"
        log_warn "Ollama will fall back to CPU mode"
    fi
    
    # Start Ollama in background
    log_info "Starting IPEX-LLM Ollama server..."
    
    # Create log directory
    mkdir -p "$SCRIPT_DIR/logs"
    
    # Start ollama serve with environment variables (setsid to fully detach)
    cd "$IPEX_OLLAMA_DIR"
    setsid env \
        LD_LIBRARY_PATH="$IPEX_OLLAMA_DIR:${LD_LIBRARY_PATH:-}" \
        SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 \
        OLLAMA_NUM_GPU=999 \
        OLLAMA_HOST=0.0.0.0:11434 \
        OLLAMA_MODELS="$OLLAMA_MODELS" \
        ./ollama serve > "$SCRIPT_DIR/logs/ipex-ollama.log" 2>&1 &
    cd "$SCRIPT_DIR"
    
    OLLAMA_PID=$!
    echo $OLLAMA_PID > "$IPEX_OLLAMA_DIR/ollama.pid"
    
    # Wait for Ollama to be ready
    log_info "Waiting for IPEX-LLM Ollama to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:11434/api/tags &>/dev/null; then
            log_info "IPEX-LLM Ollama is ready (Intel GPU mode)"
            return 0
        fi
        if [ $i -eq 60 ]; then
            log_error "IPEX-LLM Ollama failed to start within 60 seconds"
            log_error "Check logs at: $SCRIPT_DIR/logs/ipex-ollama.log"
            exit 1
        fi
        sleep 1
    done
}

# ============================================================================
# Main Startup
# ============================================================================

log_info "Starting Paperlinse..."

# Function to check and install system dependencies
check_deps() {
    local missing=()
    
    # Check for python3-venv
    if ! python3 -c "import ensurepip" 2>/dev/null; then
        missing+=("python3.11-venv")
    fi
    
    # Check for npm/nodejs
    if ! command -v npm &>/dev/null; then
        missing+=("nodejs" "npm")
    fi
    
    # Check for cifs-utils (mount.cifs)
    if ! command -v mount.cifs &>/dev/null; then
        missing+=("cifs-utils")
    fi
    
    # Install missing packages if any
    if [ ${#missing[@]} -gt 0 ]; then
        log_info "Installing missing dependencies: ${missing[*]}"
        apt update -qq
        apt install -y "${missing[@]}"
    fi
}

# Check and install dependencies
check_deps

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install backend dependencies (quietly if already installed)
pip install -q -r backend/requirements.txt

# Build frontend if needed (check for actual frontend files, not /dist subfolder)
if [ ! -f "backend/frontend/index.html" ]; then
    log_info "Building frontend..."
    cd frontend
    npm install --silent
    npm run build
    cd ..
    
    # Copy built frontend to backend
    rm -rf backend/frontend
    cp -r frontend/dist backend/frontend
fi

# Kill any existing instance
if pgrep -f "python.*main\.py" > /dev/null; then
    log_info "Stopping existing Paperlinse server..."
    pkill -f "python.*main\.py"
    sleep 1
fi

# Check for PostgreSQL
echo ""
if pg_isready -h localhost -p 5432 &>/dev/null 2>&1; then
    log_info "PostgreSQL detected on localhost:5432"
else
    log_error "PostgreSQL not detected on localhost:5432"
    log_error "Please ensure PostgreSQL is running:"
    log_error "  systemctl start postgresql"
    exit 1
fi

# Handle Ollama based on Intel GPU flag
if [ "$USE_INTEL_GPU" = true ]; then
    log_info "Setting up IPEX-LLM Ollama for Intel GPU..."
    
    # Install Intel GPU compute runtime if needed
    install_intel_gpu_deps
    
    # Install IPEX-LLM Ollama if not present
    install_ipex_ollama
    
    # Start IPEX-LLM Ollama on host (model will be pulled by the backend as needed)
    start_ipex_ollama
else
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        log_info "External Ollama detected on localhost:11434"
    else
        log_warn "Ollama not detected - LLM features may not work"
        log_warn "Install Ollama or ensure Intel GPU is available for auto-detection"
    fi
fi
echo ""

# Export database URL for the backend
export DATABASE_URL="postgresql://paperlinse:paperlinse@localhost:5432/paperlinse"

# Start the server (nohup + disown to fully detach from terminal)
# Disable oneDNN for PaddlePaddle to avoid conflicts with Intel GPU environment
log_info "Starting Paperlinse server..."
nohup env \
    PATH="$SCRIPT_DIR/venv/bin:$PATH" \
    DATABASE_URL="$DATABASE_URL" \
    FLAGS_use_mkldnn=0 \
    DNNL_VERBOSE=0 \
    "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/backend/main.py" > "$SCRIPT_DIR/logs/paperlinse.log" 2>&1 &
disown
SERVER_PID=$!
echo $SERVER_PID > "$SCRIPT_DIR/.server.pid"
cd "$SCRIPT_DIR"

# Wait for server to be ready
log_info "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/health &>/dev/null || curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ 2>/dev/null | grep -q "200\|304"; then
        break
    fi
    # Check if process is still running (use pgrep since setsid creates new session)
    if ! pgrep -f "python.*main\.py" > /dev/null; then
        log_error "Server process died unexpectedly"
        log_error "Check logs: tail -f $SCRIPT_DIR/logs/paperlinse.log"
        exit 1
    fi
    if [ $i -eq 30 ]; then
        log_warn "Server taking longer than expected to start..."
    fi
    sleep 1
done

# Final status
echo ""
log_info "Paperlinse is running!"
echo ""
echo "  Web UI:        http://localhost:8000"
echo "  PostgreSQL:    localhost:5432"
if [ "$USE_INTEL_GPU" = true ]; then
echo "  Ollama API:    http://localhost:11434 (IPEX-LLM, Intel GPU)"
fi
echo ""
echo "  LLM models can be configured in Settings > LLM Settings"
echo ""
echo "Logs:"
echo "  Server:   tail -f $SCRIPT_DIR/logs/paperlinse.log"
if [ "$USE_INTEL_GPU" = true ]; then
echo "  Ollama:   tail -f $SCRIPT_DIR/logs/ipex-ollama.log"
fi
echo ""
echo "To stop:    ./stop.sh"
if [ "$USE_INTEL_GPU" = true ]; then
echo "            ./stop.sh --intel-gpu  (also stops Ollama)"
fi
echo ""
