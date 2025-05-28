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

    fun loadLabelsFromAppendedZip(context: Context, modelPath: String): List<String>? {
        val assetManager = context.assets
        var labelsFound: List<String>? = null

        // Try with .tflite extension first, then original path as fallback
        val pathsToTry = listOf(
            YOLOUtils.ensureTFLiteExtension(modelPath), // Path with .tflite (if not already present)
            modelPath                                  // Original path
        ).distinct() // Avoid trying the same path twice if modelPath already ends with .tflite

        for (currentPathAttempt in pathsToTry) {
            var resolvedPath = currentPathAttempt
            if (resolvedPath.startsWith("flutter_assets/")) {
                resolvedPath = resolvedPath.substring("flutter_assets/".length)
            }
            Log.d(TAG, "Appended ZIP: Attempting to load labels using path: $resolvedPath (original attempt: $currentPathAttempt)")

            var afd: AssetFileDescriptor? = null
        var fis: FileInputStream? = null // Corrected indentation
        var fileChannel: FileChannel? = null
        // Removed shadowed 'labelsFound' declaration from here

        try { // This try is for the current iteration of the for loop
            val fileSize: Long
            val fileStartOffset: Long

            val file = File(resolvedPath)
            if (file.isAbsolute && file.exists()) {
                Log.d(TAG, "Appended ZIP: Accessing via File path: $resolvedPath")
                fis = FileInputStream(file)
                fileChannel = fis.channel
                fileSize = fileChannel.size()
                fileStartOffset = 0L
            } else {
                Log.d(TAG, "Appended ZIP: Accessing via Asset path: $resolvedPath")
                try {
                    afd = assetManager.openFd(resolvedPath)
                    fis = FileInputStream(afd.fileDescriptor)
                    fileChannel = fis.channel
                    fileSize = afd.declaredLength
                    fileStartOffset = afd.startOffset
                } catch (e: IOException) {
                    Log.w(TAG, "Appended ZIP: Failed to open AssetFileDescriptor for: $resolvedPath (attempt: $currentPathAttempt)", e)
                    // Don't return null immediately, try the next path in pathsToTry
                    continue // to next path in pathsToTry
                }
            }

            if (fileChannel == null || fileSize <= 4) {
                Log.w(TAG, "Appended ZIP: Invalid file size or channel for $resolvedPath (attempt: $currentPathAttempt)")
                // Don't return null immediately, try the next path in pathsToTry
                // Logged warning. 'continue' will go to 'finally' then next iteration.
                continue
            }
            Log.d(TAG, "Appended ZIP: Path=$resolvedPath (attempt: $currentPathAttempt), Size=$fileSize, StartOffset=$fileStartOffset")

            val tailSearchSize = (65536 + 256).toLong() // End of Central Directory Record is typically small
            val readLength = kotlin.math.min(fileSize, tailSearchSize)
            val readOffsetInFile = fileSize - readLength
            val channelPositionForTail = fileStartOffset + readOffsetInFile

            Log.d(TAG, "Appended ZIP: Reading tail. Channel Abs Position: $channelPositionForTail, Length: $readLength")

            val tailByteBuffer = ByteBuffer.allocate(readLength.toInt())
            fileChannel.position(channelPositionForTail)
            val totalBytesReadTail = fileChannel.read(tailByteBuffer)

            if (totalBytesReadTail <= 4) {
                Log.w(TAG, "Appended ZIP: Could not read enough bytes from the tail section ($totalBytesReadTail bytes) for $resolvedPath (attempt: $currentPathAttempt).")
                // Don't return null immediately, try the next path in pathsToTry
                // Logged warning. 'continue' will go to 'finally' then next iteration.
                continue
            }
            tailByteBuffer.flip()

            val pkHeaderSignatureLE = 0x04034B50 // Little Endian PK\03\04
            var lastPkIndexInBuffer = -1
            tailByteBuffer.order(ByteOrder.LITTLE_ENDIAN)
            for (i in totalBytesReadTail - 4 downTo 0) {
                if (tailByteBuffer.getInt(i) == pkHeaderSignatureLE) {
                    lastPkIndexInBuffer = i
                    break
                }
            }

            if (lastPkIndexInBuffer == -1) {
                Log.w(TAG, "Appended ZIP: PK\\03\\04 signature not found in the tail $totalBytesReadTail bytes of $resolvedPath (attempt: $currentPathAttempt)")
                // Don't return null immediately, try the next path in pathsToTry
                // Logged warning. 'continue' will go to 'finally' then next iteration.
                continue
            }

            val pkAbsoluteOffset = fileStartOffset + readOffsetInFile + lastPkIndexInBuffer
            Log.d(TAG, "Appended ZIP: Found last PK header signature at absolute offset in channel: $pkAbsoluteOffset")
            
            fileChannel.position(pkAbsoluteOffset) // Position channel to the start of ZIP data

            Channels.newInputStream(fileChannel).use { channelIs ->
                BufferedInputStream(channelIs).use { bis ->
                    ZipInputStream(bis).use { zis ->
                        Log.d(TAG, "Appended ZIP: Reading entries from positioned stream...")
                        var entry: ZipEntry?
                        while (zis.nextEntry.also { entry = it } != null) {
                            Log.v(TAG, "Appended ZIP: Found entry: $entry")
                            val entryName = entry!!.name
                            Log.v(TAG, "Appended ZIP: Found entry: $entryName")
                            if (entry!!.isDirectory) continue

                            if (entryName == "TFLITE_ULTRALYTICS_METADATA.json" || entryName == "metadata.json") {
                                Log.i(TAG, "Appended ZIP: Found metadata file in ZIP: $entryName")
                                val entryBos = ByteArrayOutputStream()
                                val buffer = ByteArray(4096)
                                var len: Int
                                while (zis.read(buffer).also { len = it } > 0) {
                                    entryBos.write(buffer, 0, len)
                                }
                                // zis.closeEntry() // Not strictly necessary as getNextEntry or close will handle it

                                try {
                                    val meta = JSONObject(String(entryBos.toByteArray(), StandardCharsets.UTF_8))
                                    if (!meta.has("names")) {
                                        Log.w(TAG, "Appended ZIP: Metadata file '$entryName' has no 'names'.")
                                        continue
                                    }
                                    val namesObj = meta.getJSONObject("names")
                                    val tempLabels = mutableListOf<String>()
                                    val keys = namesObj.keys().asSequence().toList()
                                    // Sort keys numerically if possible, otherwise alphabetically
                                    val sortedKeys = try {
                                        keys.sortedBy { it.toInt() }
                                    } catch (e: NumberFormatException) {
                                        keys.sorted()
                                    }
                                    for (key in sortedKeys) {
                                        tempLabels.add(namesObj.getString(key))
                                    }
                                    if (tempLabels.isNotEmpty()) {
                                        Log.i(TAG, "Labels loaded (Appended ZIP): ${tempLabels.size}")
                                        labelsFound = tempLabels
                                        // Optionally parse "date" for gpu_compatible here if needed
                                        break // Found metadata, exit loop
                                    }
                                } catch (e: org.json.JSONException) {
                                    Log.e(TAG, "Appended ZIP: Failed to parse JSON from '$entryName'", e)
                                }
                            }
                        }
                    }
                }
            }
            if (labelsFound != null) {
                Log.i(TAG, "Appended ZIP: Successfully loaded labels using path: $resolvedPath (attempt: $currentPathAttempt)")
                break // Labels found, exit the loop
            } else {
                Log.w(TAG, "Appended ZIP: Metadata JSON file not found or failed to parse for $resolvedPath (attempt: $currentPathAttempt).")
            }
            // Removed explicit closeResources call from here, 'finally' will handle it.
        } catch (e: IOException) {
            Log.w(TAG, "Appended ZIP: IOException during attempt for $resolvedPath (original attempt: $currentPathAttempt)", e)
            // Continue to the next path attempt, 'finally' will clean up.
        } catch (e: Exception) {
            Log.e(TAG, "Appended ZIP: General error during attempt for $resolvedPath (original attempt: $currentPathAttempt)", e)
            // Continue to the next path attempt, 'finally' will clean up.
        } finally {
            closeResources(afd, fis, fileChannel, "End of attempt for $resolvedPath (attempt: $currentPathAttempt)")
        }
        // If labelsFound became non-null due to a 'break' inside 'try', the loop will terminate.
    } // End of for loop pathsToTry

    if (labelsFound == null) {
        Log.w(TAG, "Appended ZIP: Failed to load labels after trying all path variants for original: $modelPath")
    }
    return labelsFound
} // End of loadLabelsFromAppendedZip function

private fun closeResources(afd: AssetFileDescriptor?, fis: FileInputStream?, fileChannel: FileChannel?, reason: String) {
    Log.d(TAG, "Appended ZIP: Closing resources ($reason).")
    try { fileChannel?.close() } catch (e: IOException) { Log.e(TAG, "Error closing FileChannel", e) }
    try { fis?.close() } catch (e: IOException) { Log.e(TAG, "Error closing FileInputStream", e) }
    try { afd?.close() } catch (e: IOException) { Log.e(TAG, "Error closing AssetFileDescriptor", e) }
}
} // End of YoloFileUtils object