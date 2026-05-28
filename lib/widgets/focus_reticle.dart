// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';

/// Animated tap-to-focus reticle. Sits in the parent `Stack`; whenever [position] flips to a new non-null offset it
/// fades in (100 ms) then back out (300 ms). Renders the plugin's `assets/focus_reticle.png`.
class FocusReticle extends StatefulWidget {
  /// View-relative pixel coordinate to render the reticle at. `null` hides it.
  final Offset? position;

  /// Reticle render size in logical pixels.
  final double size;

  const FocusReticle({super.key, required this.position, this.size = 80});

  @override
  State<FocusReticle> createState() => _FocusReticleState();
}

class _FocusReticleState extends State<FocusReticle> {
  double _opacity = 0;
  Offset? _lastPosition;

  @override
  void didUpdateWidget(FocusReticle oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Only pulse on a *new* position — re-renders with the same offset (e.g. parent setState during the fade) must not
    // retrigger the animation.
    final next = widget.position;
    if (next != null && next != _lastPosition) {
      _lastPosition = next;
      _pulse();
    }
  }

  Future<void> _pulse() async {
    setState(() => _opacity = 1);
    // Wait for the fade-in (100 ms) before kicking off the fade-out so the user actually sees the reticle peak before
    // it disappears.
    await Future<void>.delayed(const Duration(milliseconds: 400));
    if (!mounted) return;
    setState(() => _opacity = 0);
  }

  @override
  Widget build(BuildContext context) {
    final pos = _lastPosition;
    if (pos == null) return const SizedBox.shrink();
    final half = widget.size / 2;
    return Positioned(
      left: pos.dx - half,
      top: pos.dy - half,
      width: widget.size,
      height: widget.size,
      child: IgnorePointer(
        child: AnimatedOpacity(
          opacity: _opacity,
          duration: Duration(milliseconds: _opacity == 1 ? 100 : 300),
          child: Image.asset(
            'assets/focus_reticle.png',
            package: 'ultralytics_yolo',
            fit: BoxFit.contain,
          ),
        ),
      ),
    );
  }
}
