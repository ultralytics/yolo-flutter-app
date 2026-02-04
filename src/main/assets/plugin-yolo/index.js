/**
 * YOLO 插件 JavaScript 接口定义
 * 遵循 Rhino 模块规范
 * @author Ultralytics
 */

module.exports = function (plugin) {
    // 定义核心实现对象
    var yolo = {};

    /**
     * 加载模型
     * @param {string} modelPath 模型路径 (可以是绝对路径，或 assets 下的文件名)
     * @param {boolean} [useGpu=true] 是否使用 GPU 加速
     */
    yolo.loadModel = function (modelPath, useGpu) {
        if (useGpu === undefined) useGpu = true;
        plugin.loadModel(modelPath, useGpu);
    };

    /**
     * 执行检测
     * @param {ImageWrapper|Bitmap} image 图片对象
     * @returns {YOLOResult} 检测结果
     */
    yolo.detect = function (image) {
        // 适配 Auto.js 的 ImageWrapper 对象
        if (image && image.getBitmap) {
            return plugin.detect(image.getBitmap());
        }
        return plugin.detect(image);
    };

    /**
     *Base64 图片检测
     * @param {string} base64字符串
     */
    yolo.detectBase64 = function (base64) {
        return plugin.detectBase64(base64);
    };

    /**
     * 设置置信度阈值
     * @param {float} conf (0.0 - 1.0)
     */
    yolo.setConfidence = function (conf) {
        plugin.setConfidenceThreshold(conf);
    };

    /**
     * 设置 IoU 阈值
     * @param {float} iou (0.0 - 1.0)
     */
    yolo.setIou = function (iou) {
        plugin.setIouThreshold(iou);
    };

    // 底部统一导出
    return yolo;
};
