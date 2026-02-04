package com.ultralytics.yolo

import android.content.Context
import org.autojs.plugin.sdk.Plugin
import org.autojs.plugin.sdk.PluginLoader
import org.autojs.plugin.sdk.PluginRegistry

/**
 * 使用标准静态初始化块进行插件注册
 * 这是 Auto.js Plugin SDK 的官方推荐方式
 */
class YOLOPluginRegistry : PluginRegistry() {
    
    companion object {
        init {
            registerDefaultPlugin(object : PluginLoader {
                override fun load(context: Context, selfContext: Context, runtime: Any, topLevelScope: Any): Plugin? {
                    return try {
                        val plugin = YOLOPlugin(context, selfContext, runtime, topLevelScope)
                        plugin
                    } catch (e: Throwable) {
                        null
                    }
                }
            })
        }
    }
}
