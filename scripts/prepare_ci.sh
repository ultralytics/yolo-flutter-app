#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Script to prepare CI environment for dart analysis

set -e

echo "=== Preparing CI environment ==="

# Install dependencies for main package
echo "Installing dependencies for main package..."
flutter pub get

# Install dependencies for example app
echo "Installing dependencies for example app..."
cd example && flutter pub get && cd ..

# Install dependencies for demo_app
echo "Installing dependencies for demo_app..."
cd demo_app && flutter pub get && cd ..

# Install dependencies for all sample apps
echo "Installing dependencies for sample apps..."
for sample in samples/*; do
  if [ -d "$sample" ] && [ -f "$sample/pubspec.yaml" ]; then
    echo "Processing $sample..."
    cd "$sample" && flutter pub get && cd ../..
  fi
done

echo "=== CI environment preparation complete ==="
