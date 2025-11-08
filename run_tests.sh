#!/bin/bash
# Quick test runner script for EmoRec test suite

echo "🧪 EmoRec Test Suite Runner"
echo "=========================="
echo ""

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    uv pip install pytest pytest-asyncio
fi

# Function to run tests with nice output
run_tests() {
    local test_file=$1
    local description=$2
    echo "📋 Running: $description"
    uv run pytest "$test_file" -v --tb=short
    echo ""
}

# Parse command line argument
case "${1:-all}" in
    "schemas")
        run_tests "tests/test_schemas.py" "Pydantic Schema Tests"
        ;;
    "embedding")
        run_tests "tests/test_embedding_service.py" "Embedding Service Tests"
        ;;
    "emotion")
        run_tests "tests/test_emotion_mapper.py" "Emotion Mapper Tests"
        ;;
    "spotify")
        run_tests "tests/test_spotify_service.py" "Spotify Service Tests"
        ;;
    "playlist")
        run_tests "tests/test_playlist_generator.py" "Playlist Generator Tests"
        ;;
    "api")
        run_tests "tests/test_api_routes.py" "API Routes Tests"
        ;;
    "passing")
        echo "✅ Running only fully passing test suites..."
        run_tests "tests/test_schemas.py" "Schemas (31 tests)"
        run_tests "tests/test_embedding_service.py" "Embedding Service (19 tests)"
        run_tests "tests/test_playlist_generator.py" "Playlist Generator (22 tests)"
        ;;
    "integration")
        echo "🔗 Running real integration tests..."
        uv run pytest tests/test_integration.py -v
        ;;
    "unit")
        echo "🔬 Running all unit tests..."
        uv run pytest tests/ --ignore=tests/test_api_routes.py --ignore=tests/test_integration.py -v
        ;;
    "quick")
        echo "⚡ Running quick tests (no model loading)..."
        uv run pytest tests/test_schemas.py -v
        ;;
    "coverage")
        echo "📊 Running tests with coverage report..."
        uv run pytest tests/ --cov=backend --cov-report=html --cov-report=term
        echo ""
        echo "📈 Coverage report generated at: htmlcov/index.html"
        ;;
    "all")
        echo "🚀 Running all tests..."
        uv run pytest tests/ -v --tb=short
        ;;
    "help")
        echo "Usage: ./run_tests.sh [option]"
        echo ""
        echo "Options:"
        echo "  all          - Run all tests (default)"
        echo "  passing      - Run only fully passing test suites"
        echo "  integration  - Run real integration tests (no mocks)"
        echo "  unit         - Run unit tests only (with mocks)"
        echo "  quick        - Run fast schema tests only"
        echo "  coverage     - Run with coverage report"
        echo "  schemas      - Run schema validation tests"
        echo "  embedding    - Run embedding service tests"
        echo "  emotion      - Run emotion mapper tests"
        echo "  spotify      - Run Spotify service tests"
        echo "  playlist     - Run playlist generator tests"
        echo "  api          - Run API integration tests"
        echo "  help         - Show this help message"
        ;;
    *)
        echo "❌ Unknown option: $1"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

echo "✨ Test run complete!"
