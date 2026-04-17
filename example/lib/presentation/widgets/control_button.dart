// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// A circular control button that displays an icon, asset image, or text.
class ControlButton extends StatelessWidget {
  const ControlButton.icon({
    super.key,
    required IconData icon,
    required this.onPressed,
  }) : _icon = icon,
       _assetPath = null,
       _label = null;

  const ControlButton.asset({
    super.key,
    required String assetPath,
    required this.onPressed,
  }) : _icon = null,
       _assetPath = assetPath,
       _label = null;

  const ControlButton.text({
    super.key,
    required String label,
    required this.onPressed,
  }) : _icon = null,
       _assetPath = null,
       _label = label;

  final IconData? _icon;
  final String? _assetPath;
  final String? _label;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return CircleAvatar(
      radius: 24,
      backgroundColor: Colors.black.withValues(alpha: 0.2),
      child: _buildContent(),
    );
  }

  Widget _buildContent() {
    final icon = _icon;
    if (icon != null) {
      return IconButton(
        icon: Icon(icon, color: Colors.white),
        onPressed: onPressed,
      );
    }
    final assetPath = _assetPath;
    if (assetPath != null) {
      return IconButton(
        icon: Image.asset(
          assetPath,
          width: 24,
          height: 24,
          color: Colors.white,
        ),
        onPressed: onPressed,
      );
    }
    return TextButton(
      onPressed: onPressed,
      child: Text(
        _label ?? '',
        style: const TextStyle(color: Colors.white, fontSize: 12),
      ),
    );
  }
}
