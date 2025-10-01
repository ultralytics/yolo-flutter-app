// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// A circular control button that can display an icon, image, or text
class ControlButton extends StatelessWidget {
  const ControlButton({
    super.key,
    required this.content,
    required this.onPressed,
  });

  final dynamic content;
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
    if (content is IconData) {
      return IconButton(
        icon: Icon(content, color: Colors.white),
        onPressed: onPressed,
      );
    } else if (content.toString().contains('assets/')) {
      return IconButton(
        icon: Image.asset(content, width: 24, height: 24, color: Colors.white),
        onPressed: onPressed,
      );
    } else {
      return TextButton(
        onPressed: onPressed,
        child: Text(
          content,
          style: const TextStyle(color: Colors.white, fontSize: 12),
        ),
      );
    }
  }
}
