module.exports = function (plugin) {
    var yolo = {};

    yolo.loadModel = function(modelPath, useGpu) {
        if (useGpu === undefined) useGpu = true;
        plugin.loadModel(modelPath, useGpu);
    };

    yolo.detect = function(image) {
        // Automatically handle Auto.js Image wrapper if passed
        if (image && image.getBitmap) {
            return plugin.detect(image.getBitmap());
        }
        return plugin.detect(image);
    };
    
    yolo.detectBase64 = function(base64) {
        return plugin.detectBase64(base64);
    };

    yolo.setConfidence = function(conf) {
        plugin.setConfidenceThreshold(conf);
    };
    
    yolo.setIou = function(iou) {
        plugin.setIouThreshold(iou);
    };

    return yolo;
};
