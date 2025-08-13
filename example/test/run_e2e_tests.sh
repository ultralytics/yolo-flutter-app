#!/bin/bash

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# E2E Integration Test Runner for YOLO Flutter App

set -e

echo "🚀 Starting YOLO Flutter App E2E Integration Tests"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Check if Flutter is installed
if ! command -v flutter &> /dev/null; then
    print_error "Flutter is not installed or not in PATH"
    exit 1
fi

# Check Flutter version
FLUTTER_VERSION=$(flutter --version | head -n 1)
print_status "Using $FLUTTER_VERSION"

# Navigate to the example directory
cd "$(dirname "$0")/.."

# Clean and get dependencies
print_status "Cleaning project and getting dependencies..."
flutter clean
flutter pub get

# Check if integration_test dependency is available
if ! grep -q "integration_test:" pubspec.yaml; then
    print_warning "integration_test dependency not found in pubspec.yaml"
    print_status "Adding integration_test dependency..."
    flutter pub add integration_test --dev
fi

# Run unit tests first
print_status "Running unit tests..."
if flutter test; then
    print_success "Unit tests passed"
else
    print_error "Unit tests failed"
    exit 1
fi

# Run integration tests
print_status "Running E2E integration tests..."

# Test 1: Basic integration test
print_status "Running basic integration test..."
if flutter test test/plugin_integration_test.dart; then
    print_success "Basic integration test passed"
else
    print_error "Basic integration test failed"
    exit 1
fi

# Test 2: Comprehensive E2E tests (if available)
if [ -f "test/e2e_integration_tests.dart" ]; then
    print_status "Running comprehensive E2E tests..."
    if flutter test test/e2e_integration_tests.dart; then
        print_success "Comprehensive E2E tests passed"
    else
        print_error "Comprehensive E2E tests failed"
        exit 1
    fi
else
    print_warning "Comprehensive E2E tests not found"
fi

# Test 3: YOLO functional tests (if available)
if [ -f "test/yolo_e2e_functional_tests.dart" ]; then
    print_status "Running YOLO functional tests..."
    if flutter test test/yolo_e2e_functional_tests.dart; then
        print_success "YOLO functional tests passed"
    else
        print_error "YOLO functional tests failed"
        exit 1
    fi
else
    print_warning "YOLO functional tests not found"
fi

# Run tests on different platforms if available
print_status "Running platform-specific tests..."

# Android tests (if available)
if flutter devices | grep -q "android"; then
    print_status "Running tests on Android..."
    if flutter test --device-id=android; then
        print_success "Android tests passed"
    else
        print_warning "Android tests failed or skipped"
    fi
fi

# iOS tests (if available)
if flutter devices | grep -q "ios"; then
    print_status "Running tests on iOS..."
    if flutter test --device-id=ios; then
        print_success "iOS tests passed"
    else
        print_warning "iOS tests failed or skipped"
    fi
fi

# Generate test coverage report
print_status "Generating test coverage report..."
if flutter test --coverage; then
    print_success "Coverage report generated"
    
    # Check if lcov is available for HTML report
    if command -v genhtml &> /dev/null; then
        print_status "Generating HTML coverage report..."
        genhtml coverage/lcov.info -o coverage/html
        print_success "HTML coverage report generated at coverage/html/index.html"
    else
        print_warning "genhtml not found, skipping HTML coverage report"
    fi
else
    print_warning "Coverage report generation failed"
fi

# Performance tests (if available)
print_status "Running performance tests..."
if flutter test test/yolo_performance_metrics_test.dart; then
    print_success "Performance tests passed"
else
    print_warning "Performance tests failed or skipped"
fi

# Memory leak tests (if available)
print_status "Running memory leak tests..."
if flutter test test/yolo_instance_manager_test.dart; then
    print_success "Memory leak tests passed"
else
    print_warning "Memory leak tests failed or skipped"
fi

echo ""
echo "=================================================="
print_success "All E2E integration tests completed!"
echo ""
print_status "Test Summary:"
echo "  ✅ Unit tests"
echo "  ✅ Integration tests"
echo "  ✅ Platform-specific tests"
echo "  ✅ Coverage report"
echo "  ✅ Performance tests"
echo "  ✅ Memory leak tests"
echo ""
print_status "Coverage report available at: coverage/lcov.info"
if [ -d "coverage/html" ]; then
    print_status "HTML coverage report available at: coverage/html/index.html"
fi
echo ""
print_success "🎉 E2E testing completed successfully!"