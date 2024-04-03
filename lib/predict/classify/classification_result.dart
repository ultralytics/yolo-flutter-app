/// A result of an image classification.
class ClassificationResult {
  /// Creates a [ClassificationResult].
  ClassificationResult({
    required this.index,
    required this.label,
    required this.confidence,
  });

  /// Creates a [ClassificationResult] from a [json] object.
  factory ClassificationResult.fromJson(Map<String, dynamic> json) =>
      ClassificationResult(
        index: json['index'] as int,
        label: json['label'] as String,
        confidence: json['confidence'] as double,
      );

  /// The index of the label.
  final int index;

  /// The label of the classification.
  final String label;

  /// The confidence of the classification.
  final double confidence;
}
