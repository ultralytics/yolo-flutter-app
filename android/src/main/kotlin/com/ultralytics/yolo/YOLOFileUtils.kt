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

    private fun closeResources(afd: AssetFileDescriptor?, fis: FileInputStream?, fileChannel: FileChannel?, reason: String) {
        try { fileChannel?.close() } catch (e: IOException) { Log.e(TAG, "Error closing FileChannel", e) }
        try { fis?.close() } catch (e: IOException) { Log.e(TAG, "Error closing FileInputStream", e) }
        try { afd?.close() } catch (e: IOException) { Log.e(TAG, "Error closing AssetFileDescriptor", e) }
    }
} // End of YoloFileUtils object
