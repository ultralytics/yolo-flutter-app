<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Example Test Notes

This directory contains tests and helper entry points for validating the Flutter example app and the plugin integration around it.

## 🧪 What The Tests Cover

- widget behavior
- plugin integration
- example app flows
- multi-instance behavior where relevant

## 🚀 Running Tests

```bash
flutter test
```

For example-specific test entry points:

```bash
cd example
flutter test
```

## 📦 Model Assumptions

Tests should prefer:

- official model IDs when the scenario is about resolver behavior
- explicit custom asset paths when the scenario is about local asset handling

Avoid reintroducing example-only model tables or stale hardcoded model naming in new tests.
