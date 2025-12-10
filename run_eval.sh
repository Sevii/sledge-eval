#!/bin/bash

# Sledge Eval Runner with llama-server
# This script starts a llama-server instance and runs evaluations against it

set -e  # Exit on any error

# Default configuration
DEFAULT_MODEL="Qwen/Qwen3-VL-4B-Instruct-GGUF"
DEFAULT_PORT="8080"
DEFAULT_MODE="all"

# Parse command line arguments
MODEL="${1:-$DEFAULT_MODEL}"
PORT="${2:-$DEFAULT_PORT}"
MODE="${3:-$DEFAULT_MODE}"
TEST_SUITE="${4:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if llama-server is available
check_llama_server() {
    if ! command -v llama-server &> /dev/null; then
        print_error "llama-server not found. Please install llama.cpp or ensure it's in your PATH."
        print_error "You can install it with:"
        print_error "  brew install llama.cpp  # macOS"
        print_error "  # Or build from source: https://github.com/ggerganov/llama.cpp"
        exit 1
    fi
    
    local server_path=$(which llama-server)
    print_info "Found llama-server at: $server_path"
    
    # Try to get version info
    if llama-server --version &> /dev/null; then
        local version=$(llama-server --version 2>/dev/null || echo "unknown")
        print_info "llama-server version: $version"
    fi
}

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local max_attempts=60  # Increased for model loading
    local attempt=0
    
    print_info "Waiting for llama-server to be ready on port $port..."
    
    while [ $attempt -lt $max_attempts ]; do
        # First check health endpoint
        local health_response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null)
        
        if [ "$health_response" = "200" ]; then
            # Health is good, now check if models endpoint works
            local models_response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/v1/models" 2>/dev/null)
            if [ "$models_response" = "200" ]; then
                # Double check that model is actually loaded
                local models_data=$(curl -s "http://localhost:$port/v1/models" 2>/dev/null)
                if echo "$models_data" | grep -q '"object":"list"'; then
                    print_success "Server is ready and model is loaded!"
                    return 0
                else
                    echo -n "L"  # L for Loading model data
                fi
            else
                echo -n "M"  # M for Models endpoint not ready
            fi
        elif [ "$health_response" = "503" ]; then
            echo -n "H"  # H for Health check failed (server loading)
            # Wait longer when health check fails as server is still starting up
            sleep 5
            attempt=$((attempt + 2))  # Count this as 2 attempts since we waited longer
            continue
        else
            echo -n "."  # Server not responding at all
        fi
        
        attempt=$((attempt + 1))
        sleep 3
    done
    
    print_error "Server failed to be ready within $((max_attempts * 3)) seconds"
    return 1
}

# Function to stop the server
cleanup() {
    if [ ! -z "$SERVER_PID" ]; then
        print_info "Stopping llama-server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        print_success "Server stopped"
    fi
}

# Function to start the server
start_server() {
    local model=$1
    local port=$2
    
    print_info "Starting llama-server with model: $model"
    print_info "Server will be available on port: $port"
    
    # Start llama-server in background
    llama-server \
        -hf "$model" \
        --port "$port" \
        --host "127.0.0.1" \
        --ctx-size 4096 \
        --parallel 1 \
        --log-disable \
        &> llama-server.log &
    
    SERVER_PID=$!
    
    # Wait for server to be ready
    if ! wait_for_server "$port"; then
        print_error "Failed to start server. Check llama-server.log for details:"
        print_error "Last 30 lines of server log:"
        echo "----------------------------------------"
        tail -n 30 llama-server.log 2>/dev/null || echo "No log file found"
        echo "----------------------------------------"
        print_error "Common issues:"
        print_error "- Model not found or invalid format"
        print_error "- Insufficient memory"
        print_error "- Port already in use"
        print_error "- Missing dependencies"
        cleanup
        exit 1
    fi
    
    print_success "llama-server started successfully (PID: $SERVER_PID)"
}

# Function to run the evaluation
run_evaluation() {
    local mode=$1
    local port=$2
    local test_suite=$3
    
    print_info "Running evaluation in $mode mode..."
    
    # Extract model name from arguments
    local model_name_arg=""
    if [ ! -z "$MODEL" ]; then
        # Clean up model name for use as argument (replace / and : with safe characters)
        local clean_model_name=$(echo "$MODEL" | sed 's/\//_/g' | sed 's/:/_/g')
        model_name_arg="--model-name \"$MODEL\""
    fi
    
    # Build the evaluation command
    local eval_cmd="uv run eval_server.py --port $port --mode $mode"
    
    if [ ! -z "$test_suite" ]; then
        eval_cmd="$eval_cmd --test-suite $test_suite"
    fi
    
    if [ ! -z "$model_name_arg" ]; then
        eval_cmd="$eval_cmd $model_name_arg"
    fi
    
    # Run the evaluation
    print_info "Executing: $eval_cmd"
    if $eval_cmd; then
        print_success "Evaluation completed successfully"
        return 0
    else
        local exit_code=$?
        print_error "Evaluation failed with exit code: $exit_code"
        print_error "Check the output above for specific error details"
        print_error "Common issues:"
        print_error "- Server not responding to API requests"
        print_error "- Tool calling not supported by model"
        print_error "- Network connectivity issues"
        print_error "- Missing test files"
        return 1
    fi
}

# Function to display usage
usage() {
    echo "Usage: $0 [MODEL] [PORT] [MODE] [TEST_SUITE]"
    echo ""
    echo "Arguments:"
    echo "  MODEL      HuggingFace model ID (default: $DEFAULT_MODEL)"
    echo "  PORT       Server port (default: $DEFAULT_PORT)"
    echo "  MODE       Evaluation mode: single, suite, custom, all, anki (default: $DEFAULT_MODE)"
    echo "  TEST_SUITE Path to test suite JSON file (optional)"
    echo ""
    echo "Examples:"
    echo "  $0                                                    # Use defaults (all tests)"
    echo "  $0 mistralai/Ministral-8B-Instruct-2410-GGUF        # Different model, all tests"
    echo "  $0 mistralai/Ministral-3-3B-Reasoning-2512-GGUF 8081 single  # Custom port and mode"
    echo "  $0 mistralai/Ministral-3-3B-Reasoning-2512-GGUF 8080 suite tests/my_tests.json  # Custom test suite"
    echo ""
    echo "Available modes:"
    echo "  all     - Run all test types: single, suite, custom, and Anki large toolset (default)"
    echo "  single  - Run a single test case"
    echo "  suite   - Run a full test suite"
    echo "  custom  - Run with custom tool definitions"
    echo "  anki    - Run large tool set evaluation with 26+ Anki MCP tools"
}

# Main execution
main() {
    print_info "Sledge Eval Runner with llama-server"
    print_info "====================================="
    
    # Check if help was requested
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        usage
        exit 0
    fi
    
    # Validate dependencies
    check_llama_server
    
    print_info "Configuration:"
    print_info "  Model: $MODEL"
    print_info "  Port: $PORT"
    print_info "  Mode: $MODE"
    if [ ! -z "$TEST_SUITE" ]; then
        print_info "  Test Suite: $TEST_SUITE"
    fi
    echo ""
    
    # Set up cleanup trap
    trap cleanup EXIT INT TERM
    
    # Start the server
    start_server "$MODEL" "$PORT"
    
    # Run the evaluation
    if run_evaluation "$MODE" "$PORT" "$TEST_SUITE"; then
        print_success "All done! 🎉"
        exit_code=0
    else
        print_error "Evaluation failed! ❌"
        exit_code=1
    fi
    
    # Cleanup will be handled by the trap
    exit $exit_code
}

# Run main function with all arguments
main "$@"