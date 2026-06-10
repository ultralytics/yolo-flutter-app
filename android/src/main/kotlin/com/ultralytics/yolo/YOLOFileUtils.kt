// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import android.util.Log
import org.json.JSONObject
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.Channels
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

object YOLOFileUtils {
    private const val TAG = "YOLOFileUtils📁📁"

    fun loadMetadataFromAppendedZip(context: Context, modelPath: String): Map<String, Any>? {
        val assetManager = context.assets
        var metadataFound: Map<String, Any>? = null

        val pathsToTry = listOf(
            YOLOUtils.ensureTFLiteExtension(modelPath),
            modelPath
        ).distinct()

        for (currentPathAttempt in pathsToTry) {
            var resolvedPath = currentPathAttempt
            if (resolvedPath.startsWith("flutter_assets/")) {
                resolvedPath = resolvedPath.substring("flutter_assets/".length)
            }
            if (resolvedPath.startsWith("assets/")) {
                resolvedPath = resolvedPath.substring("assets/".length)
            }

            var afd: AssetFileDescriptor? = null
            var fis: FileInputStream? = null
            var fileChannel: FileChannel? = null

            try {
                val fileSize: Long
                val fileStartOffset: Long

                val file = File(resolvedPath)
                if (file.isAbsolute && file.exists()) {
                    fis = FileInputStream(file)
                    fileChannel = fis.channel
                    fileSize = fileChannel.size()
                    fileStartOffset = 0L
                } else {
                    afd = assetManager.openFd(resolvedPath)
                    fis = FileInputStream(afd.fileDescriptor)
                    fileChannel = fis.channel
                    fileSize = afd.declaredLength
                    fileStartOffset = afd.startOffset
                }

                if (fileChannel == null || fileSize <= 4) {
                    continue
                }

                val tailSearchSize = (65536 + 256).toLong()
                val readLength = kotlin.math.min(fileSize, tailSearchSize)
                val readOffsetInFile = fileSize - readLength
                val channelPositionForTail = fileStartOffset + readOffsetInFile
                val tailByteBuffer = ByteBuffer.allocate(readLength.toInt())
                fileChannel.position(channelPositionForTail)
                val totalBytesReadTail = fileChannel.read(tailByteBuffer)
                if (totalBytesReadTail <= 4) {
                    continue
                }
                tailByteBuffer.flip()

                val pkHeaderSignatureLE = 0x04034B50
                var lastPkIndexInBuffer = -1
                tailByteBuffer.order(ByteOrder.LITTLE_ENDIAN)
                for (i in totalBytesReadTail - 4 downTo 0) {
                    if (tailByteBuffer.getInt(i) == pkHeaderSignatureLE) {
                        lastPkIndexInBuffer = i
                        break
                    }
                }
                if (lastPkIndexInBuffer == -1) {
                    continue
                }

                val pkAbsoluteOffset = fileStartOffset + readOffsetInFile + lastPkIndexInBuffer
                fileChannel.position(pkAbsoluteOffset)

                Channels.newInputStream(fileChannel).use { channelIs ->
                    BufferedInputStream(channelIs).use { bis ->
                        ZipInputStream(bis).use { zis ->
                            while (true) {
                                val entry: ZipEntry = zis.nextEntry ?: break
                                if (entry.isDirectory) continue
                                val entryName = entry.name
                                if (entryName != "TFLITE_ULTRALYTICS_METADATA.json" && entryName != "metadata.json") {
                                    continue
                                }

                                val entryBos = ByteArrayOutputStream()
                                val buffer = ByteArray(4096)
                                var len: Int
                                while (zis.read(buffer).also { len = it } > 0) {
                                    entryBos.write(buffer, 0, len)
                                }

                                val metadataJson = JSONObject(String(entryBos.toByteArray(), StandardCharsets.UTF_8))
                                metadataFound = buildMap {
                                    metadataJson.optString("task").takeIf { it.isNotEmpty() }?.let { put("task", it) }
                                    metadataJson.optString("description").takeIf { it.isNotEmpty() }?.let { put("description", it) }
                                    metadataJson.optInt("stride").takeIf { it != 0 }?.let { put("stride", it) }
                                    metadataJson.optInt("channels").takeIf { it != 0 }?.let { put("channels", it) }
                                    if (metadataJson.has("end2end")) {
                                        put("end2end", metadataJson.optBoolean("end2end"))
                                    }
                                    if (metadataJson.has("imgsz")) {
                                        val imgsz = metadataJson.getJSONArray("imgsz")
                                        put(
                                            "imgsz",
                                            List(imgsz.length()) { index -> imgsz.getInt(index) }
                                        )
                                    }
                                    if (metadataJson.has("names")) {
                                        val namesObj = metadataJson.getJSONObject("names")
                                        val sortedKeys = namesObj.keys().asSequence().toList().sortedBy {
                                            it.toIntOrNull() ?: Int.MAX_VALUE
                                        }
                                        put(
                                            "labels",
                                            sortedKeys.map { key -> namesObj.getString(key) }
                                        )
                                    }
                                }
                                break
                            }
                        }
                    }
                }

                if (metadataFound != null) {
                    return metadataFound
                }
            } catch (e: Exception) {
                Log.w(TAG, "Appended ZIP: Failed to read metadata for $resolvedPath", e)
            } finally {
                closeResources(afd, fis, fileChannel, "End of metadata attempt for $resolvedPath")
            }
        }

        return null
    }

    fun loadLabelsFromAppendedZip(context: Context, modelPath: String): List<String>? {
        return (loadMetadataFromAppendedZip(context, modelPath)?.get("labels") as? List<*>)?.filterIsInstance<String>()
    }

    /**
     * Model metadata from Ultralytics' appended-ZIP, falling back to the standard embedded TFLite (FlatBuffers)
     * metadata. The fallback lets users drag-and-drop custom models that carry only standard TFLite metadata and still
     * get their task + labels auto-detected. Returns a map shaped like [loadMetadataFromAppendedZip] (keys: task,
     * labels, ...).
     */
    fun loadModelMetadata(context: Context, modelPath: String): Map<String, Any>? {
        if (modelPath.lowercase().endsWith(".onnx")) {
            return loadMetadataFromOnnx(context, modelPath) ?: loadMetadataFromSidecarYaml(context, modelPath)
        }
        loadMetadataFromAppendedZip(context, modelPath)?.let { return it }
        return loadMetadataFromFlatbuffer(context, modelPath)
    }

    /** Labels from [loadModelMetadata] (appended-ZIP or embedded FlatBuffers metadata). */
    fun loadModelLabels(context: Context, modelPath: String): List<String>? {
        return (loadModelMetadata(context, modelPath)?.get("labels") as? List<*>)?.filterIsInstance<String>()
    }

    private fun loadMetadataFromFlatbuffer(context: Context, modelPath: String): Map<String, Any>? = try {
        val buffer = YOLOUtils.loadModelFile(context, modelPath)
        val extractor = org.tensorflow.lite.support.metadata.MetadataExtractor(buffer)
        var result: Map<String, Any>? = null
        extractor.associatedFileNames?.forEach { name ->
            if (result == null) {
                extractor.getAssociatedFile(name)?.use { stream ->
                    result = parseMetadataYaml(String(stream.readBytes(), StandardCharsets.UTF_8))
                }
            }
        }
        result
    } catch (e: Exception) {
        Log.w(TAG, "Embedded FlatBuffers metadata read failed for $modelPath: ${e.message}")
        null
    }

    /**
     * Metadata embedded in an ONNX model's `metadata_props` — the Ultralytics ONNX exporter writes task, names, etc.
     * there, and QNN context-binary generation preserves them, so a `*_qnn.onnx` is fully self-contained. Only the
     * top-level protobuf fields are scanned; the large graph field is skipped, so this stays cheap for big models.
     */
    private fun loadMetadataFromOnnx(context: Context, modelPath: String): Map<String, Any>? = try {
        openModelStream(context, modelPath)?.use { stream ->
            val props = readOnnxMetadataProps(stream)
            buildMap<String, Any> {
                props["task"]?.takeIf { it.isNotEmpty() }?.let { put("task", it) }
                props["names"]?.let { names ->
                    // names is a dict-style string like "{0: 'person', 1: 'bicycle'}", valid as a YAML flow map
                    parseNames(org.yaml.snakeyaml.Yaml().load<Any?>(names))?.let { put("labels", it) }
                }
            }.ifEmpty { null }
        }
    } catch (e: Exception) {
        Log.w(TAG, "ONNX metadata_props read failed for $modelPath: ${e.message}")
        null
    }

    /** Fallback metadata for QNN ONNX exports: the `metadata.yaml` the Ultralytics exporter writes next to the model. */
    private fun loadMetadataFromSidecarYaml(context: Context, modelPath: String): Map<String, Any>? = try {
        val dir = modelPath.substringBeforeLast('/', "")
        val sidecarPath = if (dir.isEmpty()) "metadata.yaml" else "$dir/metadata.yaml"
        openModelStream(context, sidecarPath)?.use { stream ->
            parseMetadataYaml(String(stream.readBytes(), StandardCharsets.UTF_8))
        }
    } catch (e: Exception) {
        Log.w(TAG, "Sidecar metadata.yaml read failed for $modelPath: ${e.message}")
        null
    }

    /** Open [path] from the filesystem or Flutter assets (trying the usual asset prefixes). */
    private fun openModelStream(context: Context, path: String): InputStream? {
        val file = File(path)
        if (file.isAbsolute) return if (file.exists()) file.inputStream() else null
        return listOf(path, path.removePrefix("flutter_assets/"), "flutter_assets/$path").distinct()
            .firstNotNullOfOrNull { candidate ->
                try {
                    context.assets.open(candidate)
                } catch (_: IOException) {
                    null
                }
            }
    }

    /** Scan top-level ONNX ModelProto fields for `metadata_props` (field 14) key/value entries. */
    private fun readOnnxMetadataProps(stream: InputStream): Map<String, String> {
        val input = BufferedInputStream(stream)

        fun readVarint(): Long? {
            var result = 0L
            var shift = 0
            while (shift < 64) {
                val b = input.read()
                if (b < 0) return null
                result = result or ((b.toLong() and 0x7F) shl shift)
                if (b and 0x80 == 0) return result
                shift += 7
            }
            return null
        }

        fun skipFully(count: Long) {
            var remaining = count
            while (remaining > 0) {
                val skipped = input.skip(remaining)
                if (skipped > 0) remaining -= skipped else if (input.read() < 0) return else remaining--
            }
        }

        val props = mutableMapOf<String, String>()
        while (true) {
            val tag = readVarint() ?: break
            when ((tag and 7L).toInt()) {
                0 -> readVarint() ?: break
                1 -> skipFully(8)
                5 -> skipFully(4)
                2 -> {
                    val length = readVarint() ?: break
                    if ((tag ushr 3).toInt() == 14) { // metadata_props: StringStringEntryProto { key = 1; value = 2 }
                        val entry = ByteArray(length.toInt())
                        var offset = 0
                        while (offset < entry.size) {
                            val read = input.read(entry, offset, entry.size - offset)
                            if (read < 0) break
                            offset += read
                        }
                        var i = 0
                        var key: String? = null
                        var value: String? = null
                        fun entryVarint(): Long {
                            var result = 0L
                            var shift = 0
                            while (i < entry.size) {
                                val b = entry[i].toInt()
                                i++
                                result = result or ((b.toLong() and 0x7F) shl shift)
                                if (b and 0x80 == 0) break
                                shift += 7
                            }
                            return result
                        }
                        while (i < entry.size) {
                            val fieldTag = entryVarint()
                            if ((fieldTag and 7L).toInt() != 2) break
                            val textLength = entryVarint().toInt()
                            val end = i + textLength
                            if (end > entry.size || end < i) break
                            val text = String(entry, i, end - i, StandardCharsets.UTF_8)
                            i = end
                            when ((fieldTag ushr 3).toInt()) {
                                1 -> key = text
                                2 -> value = text
                            }
                        }
                        if (key != null && value != null) props[key] = value
                    } else {
                        skipFully(length)
                    }
                }
                else -> return props // unknown wire type: stop scanning
            }
        }
        return props
    }

    /** Convert a parsed `names` value (index→name map or plain list) into an ordered label list. */
    private fun parseNames(names: Any?): List<String>? = when (names) {
        is Map<*, *> -> names.entries
            .sortedBy { (k, _) -> k.toString().toIntOrNull() ?: Int.MAX_VALUE }
            .map { (_, v) -> v.toString() }
        is List<*> -> names.map { it.toString() }
        else -> null
    }

    /** Parse Ultralytics metadata YAML text into the standard metadata map shape (keys: task, labels). */
    private fun parseMetadataYaml(text: String): Map<String, Any>? {
        val parsed = org.yaml.snakeyaml.Yaml().load<Any?>(text) as? Map<*, *> ?: return null
        val map = buildMap<String, Any> {
            (parsed["task"] as? String)?.takeIf { it.isNotEmpty() }?.let { put("task", it) }
            parseNames(parsed["names"])?.let { put("labels", it) }
        }
        return map.ifEmpty { null }
    }

    private fun closeResources(afd: AssetFileDescriptor?, fis: FileInputStream?, fileChannel: FileChannel?, reason: String) {
        try { fileChannel?.close() } catch (e: IOException) { Log.e(TAG, "Error closing FileChannel", e) }
        try { fis?.close() } catch (e: IOException) { Log.e(TAG, "Error closing FileInputStream", e) }
        try { afd?.close() } catch (e: IOException) { Log.e(TAG, "Error closing AssetFileDescriptor", e) }
    }
} // End of YoloFileUtils object
