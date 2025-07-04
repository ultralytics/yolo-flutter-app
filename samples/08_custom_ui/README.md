# 08 Custom UI Sample

This sample demonstrates advanced UI customization and visualization techniques for YOLO detection results. It showcases different visual styles, animations, and interactive features to create engaging user experiences.

## Features

- ✅ Multiple visualization styles (Modern, Neon, Minimal, Glass)
- ✅ Smooth animations and transitions
- ✅ Interactive visualization options
- ✅ Confidence visualization with animated bars
- ✅ Detection grid overlay
- ✅ Heatmap visualization
- ✅ Custom drawing and effects
- ✅ Animated statistics dashboard

## Visualization Styles

### 1. Modern Style

- Rounded corners with gradient borders
- Semi-transparent fill
- Corner accent markers
- Animated scaling effect
- Confidence progress bars

### 2. Neon Style

- Glowing box effects
- Multiple blur layers
- Neon text with shadows
- Cyberpunk aesthetic
- High contrast colors

### 3. Minimal Style

- Clean corner brackets
- Simple black lines
- Minimal text labels
- No fill or effects
- Professional look

### 4. Glass Style

- Frosted glass effect
- Blur and transparency
- Soft rounded corners
- Subtle borders
- Modern iOS-like aesthetic

## Interactive Features

### Visualization Options

- **Animations**: Enable/disable all animations
- **Confidence Bar**: Show detection confidence as progress bar
- **Detection Grid**: Overlay grid for spatial reference
- **Heatmap**: Visualize detection density

### Animation Effects

```dart
// Pulse animation for floating action button
AnimationController _pulseController = AnimationController(
  duration: const Duration(seconds: 2),
  vsync: this,
)..repeat();

// Slide animation for results
AnimationController _slideController = AnimationController(
  duration: const Duration(milliseconds: 500),
  vsync: this,
);
```

## Custom Painter Implementation

The sample uses a custom painter to draw detection results:

```dart
class CustomVisualizationPainter extends CustomPainter {
  // Different drawing methods for each style
  void _drawModernStyle(Canvas canvas, Rect rect, Map<String, dynamic> detection);
  void _drawNeonStyle(Canvas canvas, Rect rect, Map<String, dynamic> detection);
  void _drawMinimalStyle(Canvas canvas, Rect rect, Map<String, dynamic> detection);
  void _drawGlassStyle(Canvas canvas, Rect rect, Map<String, dynamic> detection);
}
```

## Advanced Techniques

### 1. Gradient Borders

```dart
final borderPaint = Paint()
  ..shader = LinearGradient(
    colors: [color, color.withOpacity(0.5)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  ).createShader(rect)
  ..strokeWidth = 3
  ..style = PaintingStyle.stroke;
```

### 2. Blur Effects

```dart
// Neon glow
for (int i = 3; i > 0; i--) {
  final glowPaint = Paint()
    ..color = color.withOpacity(0.3 / i)
    ..maskFilter = MaskFilter.blur(BlurStyle.normal, i * 2.0);
  canvas.drawRect(rect, glowPaint);
}
```

### 3. Glass Morphism

```dart
// Frosted glass background
final glassPaint = Paint()
  ..color = Colors.white.withOpacity(0.1)
  ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10);
```

### 4. Animated Statistics

```dart
TweenAnimationBuilder<double>(
  tween: Tween(begin: 0, end: value),
  duration: const Duration(milliseconds: 800),
  curve: Curves.easeOutCubic,
  builder: (context, value, child) {
    // Animated progress bars
  },
);
```

## UI Components

### Style Selector

Horizontal chip list for easy style switching with visual feedback.

### Options Card

Material design card with toggle chips for visualization features.

### Detection Display

Stacked layout with image and custom painted overlay.

### Statistics Dashboard

Animated bar chart showing detection counts by class.

## Performance Considerations

1. **Animations**: Can be toggled off for better performance
2. **Blur Effects**: May impact performance on older devices
3. **Heatmap**: Computed only when enabled
4. **Grid Overlay**: Lightweight drawing operation

## Customization Ideas

- Add particle effects for detections
- Implement 3D transformation effects
- Create custom color themes
- Add sound effects for detections
- Implement gesture-based interactions
- Create AR-style overlays

## Use Cases

- 🎮 Gaming applications
- 📱 Social media filters
- 🎨 Creative tools
- 📊 Data visualization
- 🏢 Professional presentations
- 🎪 Interactive exhibitions

## Tips

- Combine multiple styles for unique effects
- Use animations sparingly for best UX
- Consider accessibility when choosing colors
- Test performance on various devices
- Adapt styles to your app's theme

## Screenshot

[Add screenshot showing different visualization styles]
