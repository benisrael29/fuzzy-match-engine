#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
VERBOSE=0
SPECIFIC_TEST=""
HEAVY_TESTS=""
COVERAGE=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -t|--test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        --heavy)
            HEAVY_TESTS="1"
            shift
            ;;
        --coverage)
            COVERAGE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose       Run tests with verbose output"
            echo "  -t, --test FILE     Run a specific test file (e.g., test_matcher.py)"
            echo "  --heavy             Run heavy dataset tests (sets RUN_HEAVY_DATASET_TESTS=1)"
            echo "  --coverage          Run tests with coverage report"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Run all tests"
            echo "  $0 -v                       # Run all tests with verbose output"
            echo "  $0 -t test_matcher          # Run specific test file"
            echo "  $0 --heavy                 # Run all tests including heavy dataset tests"
            echo "  $0 --coverage              # Run tests with coverage report"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Match Engine Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}✗ Python not found. Please install Python 3.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Using Python $PYTHON_VERSION"
echo ""

# Set environment variable for heavy tests if requested
if [ -n "$HEAVY_TESTS" ]; then
    export RUN_HEAVY_DATASET_TESTS=1
    echo -e "${YELLOW}⚠${NC} Heavy dataset tests enabled (this may take longer)"
    echo ""
fi

# Check if coverage is requested and available
if [ $COVERAGE -eq 1 ]; then
    if ! $PYTHON_CMD -m coverage --version &> /dev/null; then
        echo -e "${YELLOW}⚠${NC} Coverage not installed. Installing coverage..."
        $PYTHON_CMD -m pip install coverage --quiet
    fi
    COVERAGE_CMD="$PYTHON_CMD -m coverage run"
    COVERAGE_REPORT="$PYTHON_CMD -m coverage report"
    COVERAGE_HTML="$PYTHON_CMD -m coverage html"
else
    COVERAGE_CMD=""
    COVERAGE_REPORT=""
    COVERAGE_HTML=""
fi

# Build the test command
if [ -n "$SPECIFIC_TEST" ]; then
    # Remove .py extension if provided
    TEST_NAME=$(echo "$SPECIFIC_TEST" | sed 's/\.py$//')
    TEST_PATH="tests.${TEST_NAME}"
    echo -e "${BLUE}Running specific test: ${TEST_NAME}${NC}"
    echo ""
    
    if [ $VERBOSE -eq 1 ]; then
        TEST_CMD="$PYTHON_CMD -m unittest $TEST_PATH -v"
    else
        TEST_CMD="$PYTHON_CMD -m unittest $TEST_PATH"
    fi
else
    echo -e "${BLUE}Running all tests...${NC}"
    echo ""
    
    if [ $VERBOSE -eq 1 ]; then
        TEST_CMD="$PYTHON_CMD -m unittest discover -s tests -p 'test_*.py' -v"
    else
        TEST_CMD="$PYTHON_CMD -m unittest discover -s tests -p 'test_*.py'"
    fi
fi

# Run tests with or without coverage
if [ $COVERAGE -eq 1 ]; then
    echo -e "${BLUE}Running tests with coverage...${NC}"
    echo ""
    $COVERAGE_CMD -m unittest discover -s tests -p 'test_*.py' ${VERBOSE:+-v}
    echo ""
    echo -e "${BLUE}Coverage Report:${NC}"
    echo ""
    $COVERAGE_REPORT
    echo ""
    echo -e "${GREEN}✓${NC} Coverage report generated. Open htmlcov/index.html in a browser for detailed report."
    $COVERAGE_HTML &> /dev/null
else
    # Run the test command
    if eval "$TEST_CMD"; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}All tests passed!${NC}"
        echo -e "${GREEN}========================================${NC}"
        exit 0
    else
        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}Some tests failed!${NC}"
        echo -e "${RED}========================================${NC}"
        exit 1
    fi
fi

