# Google Play Store Deployment

This directory is the local Google Play Console staging area for the example Android app. Generated files here are
ignored by git and excluded from the pub package; only this README and the feature graphic are tracked.

Useful links:

- [Google Play Console](https://play.google.com/console)
- [Prepare and roll out a release](https://support.google.com/googleplay/android-developer/answer/9859348)
- [Upload your app to Play Console](https://developer.android.com/studio/publish/upload-bundle)
- [Inspect app versions in Latest releases and bundles](https://support.google.com/googleplay/android-developer/answer/9844279)
- [Add preview assets, including feature graphics](https://support.google.com/googleplay/android-developer/answer/9866151)

## 1. Prepare The Release

Before building store assets:

- `pubspec.yaml` must have the package version, for example `0.6.2`.
- `example/pubspec.yaml` must have the matching app version plus Android build number, for example `0.6.2+8`.
- `CHANGELOG.md` must have a top section for that version.
- Local signing files must be present:
  - `example/android/key.properties`
  - `example/android/upload/ultralytics-yolo-upload.jks`

Do not commit signing files.

## 2. Build Store Assets

Run from the repository root:

```bash
scripts/build_play_store_assets.sh
```

The script reads `pubspec.yaml` and `example/pubspec.yaml`, verifies the package version matches the example version
name, builds the signed release bundle from `example/`, verifies the AAB signature with `jarsigner`, and refreshes:

- `ultralytics-yolo-<version>-build<code>.aab`
- `ultralytics-yolo-<version>-whats-new.txt`
- `ultralytics-yolo-<version>-sha256.txt`

If `ultralytics-yolo-feature-graphic.png` is present, the script preserves it and includes it in the checksum file.

After a successful run, this directory should look like:

```text
play-store-assets/
|-- README.md
|-- ultralytics-yolo-feature-graphic.png
|-- ultralytics-yolo-<version>-build<code>.aab
|-- ultralytics-yolo-<version>-sha256.txt
`-- ultralytics-yolo-<version>-whats-new.txt
```

Only `README.md` and `ultralytics-yolo-feature-graphic.png` should be committed. The versioned files are generated
release artifacts and are ignored by git.

Use `--notes` to provide manually edited Play release notes instead of extracting the current version section from
`CHANGELOG.md`:

```bash
scripts/build_play_store_assets.sh --notes /path/to/whats-new.txt
```

## 3. Upload To Play Console

Open [Google Play Console](https://play.google.com/console), select the Ultralytics YOLO app, then create a new
release in the target track. The usual navigation is **Test and release > Production** for production releases, or the
matching testing track for internal, closed, or open testing releases.

Upload:

- App bundle: `ultralytics-yolo-<version>-build<code>.aab`
- Release notes: `ultralytics-yolo-<version>-whats-new.txt`
- Main store listing feature graphic, only when changed: `ultralytics-yolo-feature-graphic.png`

Use `ultralytics-yolo-<version>-sha256.txt` as the local record of exactly which files were produced.

For the feature graphic, use **Grow users > Store presence > Main store listing** in Play Console, then update the
feature graphic under the graphics or preview assets section.

## 4. After Submission

Review Play Console warnings before rollout. After the release is accepted, use **Test and release > Latest releases and
bundles** to confirm the uploaded version and app bundle.

Keep the generated files locally until the Play release has been accepted. They are release artifacts, not source files.

Do not commit generated AABs, generated notes, checksums, signing files, or downloaded model assets.
