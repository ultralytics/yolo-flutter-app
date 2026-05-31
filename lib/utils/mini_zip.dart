// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

/// A small ZIP extractor for standard PKZIP archives.
///
/// It supports the STORED and raw-DEFLATE methods used by Core ML
/// `.mlpackage.zip` model archives. Unsupported features are rejected rather
/// than silently mis-extracted.
class MiniZip {
  const MiniZip._();

  static const int _endOfCentralDirectorySignature = 0x06054b50;
  static const int _centralDirectorySignature = 0x02014b50;
  static const int _localFileHeaderSignature = 0x04034b50;
  static const int _zip64SizeSentinel = 0xffffffff;
  static const int _maxDeflateExpansionRatio = 1032;

  /// Extracts [bytes] into [destination].
  ///
  /// When [stripTopLevelDirectoryEndingWith] is set and every entry lives under
  /// one matching top-level directory, that directory prefix is stripped.
  static void extractBytes(
    List<int> bytes, {
    required Directory destination,
    String? stripTopLevelDirectoryEndingWith,
    bool Function(String path)? skip,
  }) {
    final data = Uint8List.fromList(bytes);
    final entries = _parseCentralDirectory(data);
    final prefix = stripTopLevelDirectoryEndingWith == null
        ? null
        : _commonTopLevelPrefix(
            entries,
            stripTopLevelDirectoryEndingWith,
            skip,
          );

    destination.createSync(recursive: true);

    for (final entry in entries) {
      var relativePath = entry.path.replaceAll('\\', '/');
      if (skip?.call(relativePath) == true) continue;
      if (prefix != null) {
        if (relativePath == prefix.substring(0, prefix.length - 1)) {
          continue;
        }
        if (!relativePath.startsWith(prefix)) {
          throw const MiniZipException.corruptArchive();
        }
        relativePath = relativePath.substring(prefix.length);
      }
      if (relativePath.isEmpty || skip?.call(relativePath) == true) continue;

      final outputPath = _safeOutputPath(destination, relativePath);
      if (relativePath.endsWith('/')) {
        Directory(outputPath).createSync(recursive: true);
        continue;
      }

      final outputFile = File(outputPath);
      outputFile.parent.createSync(recursive: true);
      outputFile.writeAsBytesSync(_contents(entry, data), flush: true);
    }
  }

  static String? _commonTopLevelPrefix(
    List<_MiniZipEntry> entries,
    String suffix,
    bool Function(String path)? skip,
  ) {
    final extractableEntries = entries
        .where((entry) => skip?.call(entry.path.replaceAll('\\', '/')) != true)
        .toList(growable: false);
    if (extractableEntries.isEmpty) return null;
    final firstPath = extractableEntries.first.path.replaceAll('\\', '/');
    if (!firstPath.contains('/')) return null;
    final topLevel = firstPath.split('/').first;
    if (!topLevel.endsWith(suffix)) return null;
    final prefix = '$topLevel/';
    return extractableEntries.every((entry) {
          final path = entry.path.replaceAll('\\', '/');
          return path == topLevel || path.startsWith(prefix);
        })
        ? prefix
        : null;
  }

  static List<_MiniZipEntry> _parseCentralDirectory(Uint8List data) {
    final count = data.length;
    if (count < 22) throw const MiniZipException.notAZipFile();

    final searchLimit = math.min(count, 22 + 0xffff);
    var eocd = -1;
    var i = count - 22;
    while (i >= count - searchLimit) {
      if (_u32(data, i) == _endOfCentralDirectorySignature &&
          i + 22 + _u16(data, i + 20) == count) {
        eocd = i;
        break;
      }
      i--;
    }
    if (eocd < 0) throw const MiniZipException.notAZipFile();

    final entryCount = _u16(data, eocd + 10);
    final centralDirectoryOffset = _u32(data, eocd + 16);
    if (centralDirectoryOffset > count) {
      throw const MiniZipException.corruptArchive();
    }

    final entries = <_MiniZipEntry>[];
    var p = centralDirectoryOffset;
    for (var index = 0; index < entryCount; index++) {
      if (p + 46 > count || _u32(data, p) != _centralDirectorySignature) {
        throw const MiniZipException.corruptArchive();
      }

      final flags = _u16(data, p + 8);
      final method = _u16(data, p + 10);
      final crc = _u32(data, p + 16);
      final compressedSize = _u32(data, p + 20);
      final uncompressedSize = _u32(data, p + 24);
      final nameLength = _u16(data, p + 28);
      final extraLength = _u16(data, p + 30);
      final commentLength = _u16(data, p + 32);
      final localHeaderOffset = _u32(data, p + 42);

      if (flags & 0x1 != 0) {
        throw const MiniZipException.unsupportedFeature('encryption');
      }
      if (compressedSize == _zip64SizeSentinel ||
          uncompressedSize == _zip64SizeSentinel ||
          localHeaderOffset == _zip64SizeSentinel) {
        throw const MiniZipException.unsupportedFeature('ZIP64');
      }

      final nameStart = p + 46;
      if (nameStart + nameLength > count) {
        throw const MiniZipException.corruptArchive();
      }
      final name = utf8.decode(
        data.sublist(nameStart, nameStart + nameLength),
        allowMalformed: true,
      );

      if (localHeaderOffset + 30 > count ||
          _u32(data, localHeaderOffset) != _localFileHeaderSignature) {
        throw const MiniZipException.corruptArchive();
      }
      final localNameLength = _u16(data, localHeaderOffset + 26);
      final localExtraLength = _u16(data, localHeaderOffset + 28);
      final dataOffset =
          localHeaderOffset + 30 + localNameLength + localExtraLength;
      if (dataOffset + compressedSize > count) {
        throw const MiniZipException.corruptArchive();
      }

      entries.add(
        _MiniZipEntry(
          path: name,
          compressionMethod: method,
          compressedSize: compressedSize,
          uncompressedSize: uncompressedSize,
          crc32: crc,
          dataOffset: dataOffset,
        ),
      );

      p = nameStart + nameLength + extraLength + commentLength;
    }
    return entries;
  }

  static List<int> _contents(_MiniZipEntry entry, Uint8List data) {
    final raw = data.sublist(
      entry.dataOffset,
      entry.dataOffset + entry.compressedSize,
    );
    final output = switch (entry.compressionMethod) {
      0 => raw,
      8 => _inflate(raw, entry.uncompressedSize, entry.path),
      _ => throw MiniZipException.unsupportedFeature(
        'compression method ${entry.compressionMethod}',
      ),
    };

    if (_crc32(output) != entry.crc32) {
      throw const MiniZipException.corruptArchive();
    }
    return output;
  }

  static List<int> _inflate(List<int> input, int expectedSize, String path) {
    if (expectedSize == 0) return const <int>[];
    if (expectedSize > input.length * _maxDeflateExpansionRatio) {
      throw MiniZipException.entryTooLarge(path);
    }

    final output = ZLibDecoder(raw: true).convert(input);
    if (output.length != expectedSize) {
      throw MiniZipException.inflateFailed(path);
    }
    return output;
  }

  static String _safeOutputPath(Directory root, String relativePath) {
    if (relativePath.startsWith('/') ||
        relativePath.startsWith(r'\') ||
        RegExp(r'^[A-Za-z]:').hasMatch(relativePath)) {
      throw MiniZipException.unsafePath(relativePath);
    }

    final segments = relativePath
        .split('/')
        .where((segment) => segment.isNotEmpty && segment != '.')
        .toList(growable: false);
    if (segments.any((segment) => segment == '..')) {
      throw MiniZipException.unsafePath(relativePath);
    }

    final separator = Platform.pathSeparator;
    return <String>[root.path, ...segments].join(separator);
  }

  static int _u16(Uint8List data, int offset) =>
      data[offset] | (data[offset + 1] << 8);

  static int _u32(Uint8List data, int offset) =>
      data[offset] |
      (data[offset + 1] << 8) |
      (data[offset + 2] << 16) |
      (data[offset + 3] << 24);

  static final List<int> _crcTable = List<int>.generate(256, (index) {
    var value = index;
    for (var bit = 0; bit < 8; bit++) {
      value = (value & 1) != 0 ? (0xedb88320 ^ (value >> 1)) : (value >> 1);
    }
    return value;
  }, growable: false);

  static int _crc32(List<int> data) {
    var crc = 0xffffffff;
    for (final byte in data) {
      crc = _crcTable[(crc ^ byte) & 0xff] ^ (crc >> 8);
    }
    return (crc ^ 0xffffffff) & 0xffffffff;
  }
}

class _MiniZipEntry {
  const _MiniZipEntry({
    required this.path,
    required this.compressionMethod,
    required this.compressedSize,
    required this.uncompressedSize,
    required this.crc32,
    required this.dataOffset,
  });

  final String path;
  final int compressionMethod;
  final int compressedSize;
  final int uncompressedSize;
  final int crc32;
  final int dataOffset;
}

class MiniZipException implements Exception {
  const MiniZipException._(this.message);

  const MiniZipException.notAZipFile() : this._('Not a valid ZIP archive.');

  const MiniZipException.corruptArchive() : this._('Corrupt ZIP archive.');

  const MiniZipException.unsupportedFeature(String feature)
    : this._('Unsupported ZIP feature: $feature.');

  const MiniZipException.inflateFailed(String path)
    : this._('Failed to inflate entry: $path.');

  const MiniZipException.unsafePath(String path)
    : this._('Refusing to extract entry outside destination: $path.');

  const MiniZipException.entryTooLarge(String path)
    : this._('Entry is too large to extract safely: $path.');

  final String message;

  @override
  String toString() => 'MiniZipException: $message';
}
