# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

Respecting these principles is critical for every PR.

**Less is more. The simplest solution is the best solution.**

The action hierarchy for every change: **Delete > Replace > Add**. The best code change is a deletion. The second best is modifying what exists. Adding new code is the last resort.

1. **Minimal**: The simplest solution that works. Do not over-engineer, over-abstract, or add code just in case. Three similar lines beat a premature abstraction. Avoid error handling for impossible states, feature flags, compatibility shims, or policy scaffolding unless they are truly required.
2. **Solve at the source**: Do not hack fixes. Solve problems at their root. If something is broken, fix or remove the broken thing. Never patch over a broken abstraction, add workarounds, or add synchronization code for state that should not be duplicated.
3. **Delete ruthlessly**: When replacing code, delete what it replaced. Remove unused imports, functions, types, files, and commented-out code. Git preserves history. Run the repo's relevant dead-code or cleanup check when available.
4. **Replace > Add**: Modify existing code over adding new code. Edit existing files, extend existing components or functions with minimal parameters, and reuse existing utilities. If creating a new file, first prove it cannot fit cleanly in an existing file.
5. **Check existing**: Search the entire repo before creating anything new. If a feature, component, helper, responder, workflow, or utility already solves a similar problem, reuse or adapt it and delete the duplicate path.
6. **Deduplicate**: Do not duplicate existing code when updating the repo. Consolidate or refactor duplicates you find when it is in scope and low risk.
7. **Zero Regression**: Do not break existing features or workflows unless the PR intentionally removes them with evidence.
8. **Production ready**: All changes must be thoroughly debugged, validated, and production ready.

**When fixing bugs, ask: "What can I delete?" before "What can I replace?" before "What should I add?"**

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Launch an independent adversarial review agent with cold context (just the PR diff and this file) to hunt for bugs, regressions, and Core Principles violations — use the Codex CLI, one fresh `codex exec` run per round. Fix, push, and repeat until a fresh run reports LGTM.
3. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never force-push, reset, or revert commits you did not author.
4. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
flutter pub get                             # install dependencies (repeat in example/ for the example app)
flutter test                                # run all tests
flutter test test/yolo_test.dart            # run one test file
flutter test --plain-name 'exact test name' # run one test by name
flutter test --coverage                     # coverage; ci.yml then strips lib/platform/, lib/yolo_view.dart, and lib/widgets/yolo_showcase.dart from coverage/lcov.info with an awk filter before Codecov upload
dart analyze --fatal-infos                  # lint gate, exactly as analyzer.yml runs it
dart format .                               # Dart formatting (format.yml enforces via Ultralytics Actions)
dart pub publish --dry-run                  # pub.dev package validation, run by ci.yml and publish.yml
```

- CI (`ci.yml`) runs three jobs on push/PR to main: `tests` (ubuntu-latest: tests + coverage + publish dry-run), `example-android` (ubuntu-latest: debug APK, API 34 emulator smoke test, release AAB verified with `scripts/build_play_store_assets.sh --verify-aab`), and `example-ios` (macos-26: SwiftPM build + simulator smoke test, then a CocoaPods regression build of the same sources).
- Version floors are Dart SDK `^3.8.1` and Flutter `>=3.32.1` (pubspec.yaml); CI uses the stable Flutter channel.
- For the Python model-export tooling use `uv pip install`, never bare `pip install` (see the header of `scripts/export-tflite-models.py`).

## Architecture

This repo is `ultralytics_yolo`, the official Ultralytics Flutter plugin for running YOLO models on Android and iOS. The Dart layer (`lib/`) exposes two entry points — `YOLO` (`lib/yolo.dart`, single-image inference) and `YOLOView` (`lib/yolo_view.dart`, real-time camera platform view) — plus `YOLOShowcase` (`lib/widgets/yolo_showcase.dart`, the full camera UI built from the exported Material widgets). Dart talks to native code over method channels defined in `lib/platform/`; per-instance channels are managed by `lib/yolo_instance_manager.dart`.

Official model IDs (e.g. `yolo26n`) are resolved by `lib/core/yolo_model_resolver.dart` to pinned GitHub release assets: Android LiteRT `.tflite` and opt-in QNN `.onnx` from yolo-flutter-app `v0.6.6`, and iOS Core ML `.mlpackage.zip` from yolo-ios-app `v8.3.0`. These tags are intentionally pinned and duplicated in `scripts/fetch_bundled_models.sh` — keep both in sync when regenerating assets (`scripts/export-tflite-models.py` for Android; the iOS assets come from the yolo-ios-app repo).

**Cross-platform model parity is mandatory:** Android and iOS must expose all 35 official YOLO26 IDs (five sizes across detect, segment, semantic, depth, classify, pose, and OBB). Never filter an official task or size by platform without explicit user approval. Any catalog change must keep the platform-independent 35-model URL test and must be validated on physical Android and iOS devices before the PR is called production-ready.

Android native code lives in `android/src/main/kotlin/com/ultralytics/yolo/` and runs LiteRT 2.x with an automatic GPU→CPU accelerator ladder, plus opt-in Snapdragon NPU support via the ONNX Runtime QNN provider (`compileOnly` dependency, so consumers opt in; see `OrtQnnModel.kt`). iOS native code in `ios/ultralytics_yolo/Sources/ultralytics_yolo/` is only the Flutter bridge and camera/view layer over the shared `UltralyticsYOLO` Swift package from yolo-ios-app (the single source of truth for iOS inference); the same source tree ships through both Swift Package Manager (`ios/ultralytics_yolo/Package.swift`) and CocoaPods (`ios/ultralytics_yolo.podspec`) — keep the two manifests in sync.

The `example/` app is the published Google Play app (`com.ultralytics.yolo`): `scripts/fetch_bundled_models.sh` bundles nano models into its assets at platform-build time (skipped in CI), `ENABLE_QNN=1` opts its build into the QNN runtime, and Play Store upload assets are built with `scripts/build_play_store_assets.sh`.

Publishing is push-triggered: `publish.yml` runs on every push to main (gated to `ultralytics/yolo-flutter-app` and actor `glenn-jocher`), compares the pubspec version against pub.dev via `ultralytics-actions`, and when it is ahead tags `vX.Y.Z` and creates a GitHub release; the tag push then fires `publish-on-tag.yml`, which does the actual `dart pub publish --force` through pub.dev trusted publishing. `tag.yml` is a manual (workflow_dispatch) tag/release fallback.

## Conventions

- Every source file opens with the `Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` header in the language's comment style; Ultralytics Actions adds them automatically — don't add or revert them manually.
- `format.yml` (Ultralytics Actions) auto-formats Dart, Swift, Python, and Prettier targets (YAML/JSON/Markdown) directly on PR branches, so `git pull --rebase` before pushing follow-up commits.
- Linting is `dart analyze --fatal-infos` against `analysis_options.yaml` (flutter_lints plus extra rules — e.g. `prefer_single_quotes`, `always_declare_return_types`, `avoid_print`).
- Dart tests in `test/` run against mocked method channels (`test/utils/test_helpers.dart`) — no live network; `example/integration_test/` holds manual on-device QNN tests that CI does not run.
- Releases: bump the version in `pubspec.yaml`, `ios/ultralytics_yolo.podspec`, and `example/pubspec.yaml` (Play Store build number) together and add a `CHANGELOG.md` entry; merging to main then auto-tags and publishes via `publish.yml`.
- `.pubignore` controls the pub.dev payload (model binaries, `play-store-assets/`, and agent docs are excluded); `dart pub publish --dry-run` in CI catches payload regressions, and `publish.yml` additionally asserts `Package.swift` ships in the archive.
