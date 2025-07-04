#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Script to analyze sample apps with appropriate strictness level

set -e

echo "=== Analyzing sample applications ==="

# Analyze each sample app
for sample in samples/*; do
  if [ -d "$sample" ] && [ -f "$sample/pubspec.yaml" ]; then
    echo "Analyzing $sample..."
    cd "$sample"
    # Run analyzer without fatal-infos for samples (only fatal-warnings)
    dart analyze --fatal-warnings || exit 1
    cd ../..
  fi
done

echo "=== Sample analysis complete ==="