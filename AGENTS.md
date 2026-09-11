# AGENTS.md

Repository guidance for coding agents. `CLAUDE.md` is a symlink to this file.

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands and validation

```bash
flutter pub get
flutter test
dart analyze --fatal-infos
dart pub publish --dry-run
g++ -std=c++17 android/src/test/cpp/depth-colorizer-test.cpp -o /tmp/depth-colorizer-test && /tmp/depth-colorizer-test
```

Run example commands from `example/` after `flutter pub get` there. CI checks both SwiftPM and CocoaPods iOS builds, Android builds, and process launch; it does not establish camera, model download, or inference correctness. Dart tests mock channels and select different resolver paths on macOS and Linux. Use a device for native inference and QNN changes. Read `.github/workflows/ci.yml` and `doc/performance.md` for native validation.

## Where to look

- Dart API and model resolution → `lib/`.
- Android inference and platform channels → `android/src/main/`.
- iOS bridge → `ios/`.
- Example and device benchmarks → `example/`.
- Asset export and packaging → `scripts/`.
- Native performance decisions → `doc/performance.md`.

## Conventions

- Every source file opens with the `Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` header in the language's comment style; Ultralytics Actions adds them automatically — don't add or revert them manually.
- `format.yml` (Ultralytics Actions) auto-formats Dart, Swift, Python, and Prettier targets (YAML/JSON/Markdown) directly on PR branches, so `git pull --rebase` before pushing follow-up commits.
- Linting is `dart analyze --fatal-infos` against `analysis_options.yaml` (flutter_lints plus extra rules — e.g. `prefer_single_quotes`, `always_declare_return_types`, `avoid_print`). The analyzer excludes `android/`, `ios/`, and generated files; `very_good_analysis` is a dev dependency but is not included by `analysis_options.yaml`.
- Dart tests in `test/` run against mocked method channels (`test/utils/test_helpers.dart`) — no live network; `example/integration_test/` holds manual on-device QNN tests that CI does not run.
- Releases: bump the version in `pubspec.yaml`, `ios/ultralytics_yolo.podspec`, and `example/pubspec.yaml` (Play Store build number) together and add a `CHANGELOG.md` entry; merging to main then auto-tags and publishes via `publish.yml`.
- `.pubignore` controls the pub.dev payload (model binaries, `play-store-assets/`, and agent docs are excluded); `dart pub publish --dry-run` in CI catches payload regressions, and `publish.yml` additionally asserts `Package.swift` ships in the archive.
- `CHANGELOG.md` entries are `## X.Y.Z` headings with `- **Fix**: ...` / `- **Feature**: ...` bullets; `scripts/build_play_store_assets.sh` extracts the current version's bullets (stripping `**` and backticks) as the Play "what's new" text and warns above 500 bytes.
- Never commit model weights: the root `.gitignore` blocks `*.tflite`, `*.mlpackage`, `*.mlpackage.zip`, `*.onnx`, `*.pt`, and friends. Official assets are GitHub release attachments, fetched at runtime by the resolver or at build time by `scripts/fetch_bundled_models.sh`.
- `YOLOTask` (`lib/models/yolo_task.dart`) defines the canonical task order `detect, segment, semantic, depth, classify, pose, obb`; `YOLOModelResolver` generates the 35 official IDs from it. Kotlin parses the wire name with `YOLOTask.valueOf(task.uppercase())`, Swift with `YOLOTask.fromString` (from the `UltralyticsYOLO` package).

## Pitfalls

- Two non-multi-instance `YOLO` objects share the native `'default'` instance. Android `putIfAbsent` and iOS "already loaded → success" mean the second `loadModel` silently keeps the first model. Use `useMultiInstance: true` (or `dispose()` first) when running two different models.
- iOS Flutter-asset models are expected as `.mlpackage.zip` archives (any `assets/...` path); the Dart resolver extracts them to documents. Other iOS asset paths pass through unchanged; `inspectModel` first looks for them via `checkModelExists` (bundled `flutter_assets`, bundle resources) and otherwise fails in `MLModel.compileModel`, which reaches Dart as a `PlatformException(MODEL_INSPECTION_FAILED)`. The "camera without inference while reporting success" branch in `YOLOView.setModel` is reached only if inspection succeeds and the camera loader still cannot locate the path. On Android, `assets/...` models are copied into the documents root (not `mobile-standard-v1/`) and never refreshed once the copy exists.
- The Android GPU program cache key is file basename + byte length, so a same-name, same-size model replacement collides on that key.
- Android release builds must keep the plugin's `android/consumer-rules.pro` (LiteRT is invoked via JNI/reflection); the example duplicates them in `example/android/app/proguard-rules.pro`.
- QNN: the runtime is `compileOnly` in the plugin, so consumers (and the example) must add `onnxruntime-android-qnn` and `useLegacyPackaging = true` themselves; the example only does so when `qnnEnabled` (`ENABLE_QNN` or `-Pqnn`). QNN models have no CPU fallback and are not resolved by model ID — pass the release URL or file path.
