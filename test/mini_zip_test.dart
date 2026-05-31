// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/utils/mini_zip.dart';

void main() {
  group('MiniZip', () {
    test('extracts deflate and stored entries with top-level stripping', () {
      final destination = Directory.systemTemp.createTempSync('mini_zip_');
      addTearDown(() => destination.deleteSync(recursive: true));

      MiniZip.extractBytes(
        _decodeBase64(_fixtureBase64),
        destination: destination,
        stripTopLevelDirectoryEndingWith: '.mlpackage',
        skip: (path) => path.startsWith('__MACOSX/') || path.contains('/._'),
      );

      expect(
        File('${destination.path}/labels.txt').readAsStringSync(),
        List.filled(24, 'hello yolo ').join(),
      );
      expect(File('${destination.path}/data/blob.bin').readAsBytesSync(), <int>[
        7,
        11,
        13,
        0,
        255,
        128,
        64,
        32,
      ]);
      expect(Directory('${destination.path}/__MACOSX').existsSync(), isFalse);
      expect(File('${destination.path}/._labels.txt').existsSync(), isFalse);
    });

    test('writes skipped metadata when no skip predicate is provided', () {
      final destination = Directory.systemTemp.createTempSync('mini_zip_');
      addTearDown(() => destination.deleteSync(recursive: true));

      MiniZip.extractBytes(
        _decodeBase64(_fixtureBase64),
        destination: destination,
      );

      expect(
        File('${destination.path}/model.mlpackage/._labels.txt').existsSync(),
        isTrue,
      );
    });

    test('rejects garbage input', () {
      final destination = Directory.systemTemp.createTempSync('mini_zip_');
      addTearDown(() => destination.deleteSync(recursive: true));

      expect(
        () => MiniZip.extractBytes(
          List<int>.filled(4096, 0xab),
          destination: destination,
        ),
        throwsA(isA<MiniZipException>()),
      );
    });

    test('rejects truncated archive', () {
      final destination = Directory.systemTemp.createTempSync('mini_zip_');
      addTearDown(() => destination.deleteSync(recursive: true));
      final full = _decodeBase64(_fixtureBase64);

      expect(
        () => MiniZip.extractBytes(
          full.sublist(0, full.length - 120),
          destination: destination,
        ),
        throwsA(isA<MiniZipException>()),
      );
    });

    test('handles archive comment containing EOCD signature bytes', () {
      final destination = Directory.systemTemp.createTempSync('mini_zip_');
      addTearDown(() => destination.deleteSync(recursive: true));

      MiniZip.extractBytes(
        base64Decode(
          'UEsDBBQAAAAIAAOav1y2t1DJDwAAAFgAAAAaAAAAbW9kZWwubWxwYWNrYWdlL2xhYmVscy50eHTLSM3JyVeozAcSGdRk'
          'AgBQSwECFAMUAAAACAADmr9ctrdQyQ8AAABYAAAAGgAAAAAAAAAAAAAAgAEAAAAAbW9kZWwubWxwYWNrYWdlL2xhYmVs'
          'cy50eHRQSwUGAAAAAAEAAQBIAAAARwAAABkAUEsFBkNPTU1FTlRQQURESU5HUEFERElORw==',
        ),
        destination: destination,
        stripTopLevelDirectoryEndingWith: '.mlpackage',
      );

      expect(
        File('${destination.path}/labels.txt').readAsStringSync(),
        List.filled(8, 'hello yolo ').join(),
      );
    });

    test('extracts empty files with CRC-32 zero', () {
      final destination = Directory.systemTemp.createTempSync('mini_zip_');
      addTearDown(() => destination.deleteSync(recursive: true));

      MiniZip.extractBytes(
        base64Decode(
          'UEsDBBQAAAAIAAOav1wAAAAAAgAAAAAAAAAZAAAAbW9kZWwubWxwYWNrYWdlL2VtcHR5LmJpbgMAUEsDBBQAAAAIAAOav1xH'
          '3dx5BAAAAAIAAAAYAAAAbW9kZWwubWxwYWNrYWdlL25vdGUudHh0y88GAFBLAQIUAxQAAAAIAAOav1wAAAAAAgAAAAAAAAAZ'
          'AAAAAAAAAAAAAACAAQAAAABtb2RlbC5tbHBhY2thZ2UvZW1wdHkuYmluUEsBAhQDFAAAAAgAA5q/XEfd3HkEAAAAAgAAABgA'
          'AAAAAAAAAAAAAIABOQAAAG1vZGVsLm1scGFja2FnZS9ub3RlLnR4dFBLBQYAAAAAAgACAI0AAABzAAAAAAA=',
        ),
        destination: destination,
        stripTopLevelDirectoryEndingWith: '.mlpackage',
      );

      expect(File('${destination.path}/empty.bin').readAsBytesSync(), isEmpty);
      expect(File('${destination.path}/note.txt').readAsStringSync(), 'ok');
    });

    test('rejects implausible deflate expansion before allocation', () {
      final destination = Directory.systemTemp.createTempSync('mini_zip_');
      addTearDown(() => destination.deleteSync(recursive: true));

      expect(
        () => MiniZip.extractBytes(
          base64Decode(
            'UEsDBBQAAAAIAAOav1zJl7hQBgAAAAQAAAAZAAAAbW9kZWwubWxwYWNrYWdlL3NtYWxsLnR4dCvJzKsEAFBLAQIUAxQAAAAI'
            'AAOav1zJl7hQBgAAAABlzR0ZAAAAAAAAAAAAAACAAQAAAABtb2RlbC5tbHBhY2thZ2Uvc21hbGwudHh0UEsFBgAAAAABAAEA'
            'RwAAAD0AAAAAAA==',
          ),
          destination: destination,
        ),
        throwsA(isA<MiniZipException>()),
      );
    });

    test('rejects entries outside the destination directory', () {
      final destination = Directory.systemTemp.createTempSync('mini_zip_');
      addTearDown(() => destination.deleteSync(recursive: true));

      expect(
        () => MiniZip.extractBytes(
          _storedZip({'../escape.txt': utf8.encode('nope')}),
          destination: destination,
        ),
        throwsA(isA<MiniZipException>()),
      );
      expect(File('${destination.path}/../escape.txt').existsSync(), isFalse);
    });
  });
}

const _fixtureBase64 = '''
UEsDBBQAAAAIAAAAIQCkHSt4DwAAAAgBAAAaAAAAbW9kZWwubWxwYWNrYWdlL2xhYmVscy50eHTL
SM3JyVeozAcSGSOZCQBQSwMEFAAAAAAAAAAhAOjjWSUIAAAACAAAAB0AAABtb2RlbC5tbHBhY2th
Z2UvZGF0YS9ibG9iLmJpbgcLDQD/gEAgUEsDBBQAAAAAAL2Vv1w5nPsGBAAAAAQAAAAlAAAAX19N
QUNPU1gvbW9kZWwubWxwYWNrYWdlLy5fbGFiZWxzLnR4dGp1bmtQSwMEFAAAAAAAvZW/XDmc+wYE
AAAABAAAABwAAABtb2RlbC5tbHBhY2thZ2UvLl9sYWJlbHMudHh0anVua1BLAQIUAxQAAAAIAAAA
IQCkHSt4DwAAAAgBAAAaAAAAAAAAAAAAAACAAQAAAABtb2RlbC5tbHBhY2thZ2UvbGFiZWxzLnR4
dFBLAQIUAxQAAAAAAAAAIQDo41klCAAAAAgAAAAdAAAAAAAAAAAAAACAAUcAAABtb2RlbC5tbHBh
Y2thZ2UvZGF0YS9ibG9iLmJpblBLAQIUAxQAAAAAAL2Vv1w5nPsGBAAAAAQAAAAlAAAAAAAAAAAA
AACAAYoAAABfX01BQ09TWC9tb2RlbC5tbHBhY2thZ2UvLl9sYWJlbHMudHh0UEsBAhQDFAAAAAAA
vZW/XDmc+wYEAAAABAAAABwAAAAAAAAAAAAAAIAB0QAAAG1vZGVsLm1scGFja2FnZS8uX2xhYmVs
cy50eHRQSwUGAAAAAAQABAAwAQAADwEAAAAA
''';

Uint8List _decodeBase64(String value) =>
    base64Decode(value.replaceAll(RegExp(r'\s+'), ''));

Uint8List _storedZip(Map<String, List<int>> files) {
  final output = BytesBuilder();
  final central = BytesBuilder();
  var offset = 0;

  for (final entry in files.entries) {
    final name = utf8.encode(entry.key);
    final data = entry.value;
    final crc = _crc32(data);

    _writeU32(output, 0x04034b50);
    _writeU16(output, 20);
    _writeU16(output, 0);
    _writeU16(output, 0);
    _writeU16(output, 0);
    _writeU16(output, 0);
    _writeU32(output, crc);
    _writeU32(output, data.length);
    _writeU32(output, data.length);
    _writeU16(output, name.length);
    _writeU16(output, 0);
    output.add(name);
    output.add(data);

    _writeU32(central, 0x02014b50);
    _writeU16(central, 20);
    _writeU16(central, 20);
    _writeU16(central, 0);
    _writeU16(central, 0);
    _writeU16(central, 0);
    _writeU16(central, 0);
    _writeU32(central, crc);
    _writeU32(central, data.length);
    _writeU32(central, data.length);
    _writeU16(central, name.length);
    _writeU16(central, 0);
    _writeU16(central, 0);
    _writeU16(central, 0);
    _writeU16(central, 0);
    _writeU32(central, 0);
    _writeU32(central, offset);
    central.add(name);

    offset = output.length;
  }

  final centralBytes = central.toBytes();
  final centralOffset = output.length;
  output.add(centralBytes);
  _writeU32(output, 0x06054b50);
  _writeU16(output, 0);
  _writeU16(output, 0);
  _writeU16(output, files.length);
  _writeU16(output, files.length);
  _writeU32(output, centralBytes.length);
  _writeU32(output, centralOffset);
  _writeU16(output, 0);
  return output.toBytes();
}

void _writeU16(BytesBuilder builder, int value) {
  builder.add(<int>[value & 0xff, (value >> 8) & 0xff]);
}

void _writeU32(BytesBuilder builder, int value) {
  builder.add(<int>[
    value & 0xff,
    (value >> 8) & 0xff,
    (value >> 16) & 0xff,
    (value >> 24) & 0xff,
  ]);
}

int _crc32(List<int> data) {
  var crc = 0xffffffff;
  for (final byte in data) {
    var value = (crc ^ byte) & 0xff;
    for (var bit = 0; bit < 8; bit++) {
      value = (value & 1) != 0 ? (0xedb88320 ^ (value >> 1)) : (value >> 1);
    }
    crc = value ^ (crc >> 8);
  }
  return (crc ^ 0xffffffff) & 0xffffffff;
}
