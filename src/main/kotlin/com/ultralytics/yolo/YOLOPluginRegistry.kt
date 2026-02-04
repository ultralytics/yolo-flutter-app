package com.ultralytics.yolo

import android.content.Context
import org.autojs.plugin.sdk.Plugin
import org.autojs.plugin.sdk.PluginLoader
import org.autojs.plugin.sdk.PluginRegistry

class YOLOPluginRegistry : PluginRegistry() {
    init {
        registerDefaultPlugin(object : PluginLoader {
            override fun load(context: Context, selfContext: Context, runtime: Any, topLevelScope: Any): Plugin {
                return YOLOPlugin(context, selfContext, runtime, topLevelScope)
            }
        })
    }
}
