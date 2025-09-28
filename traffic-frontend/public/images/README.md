# Traffic Map Background Images

## 📁 How to Add Your Local Image

1. **Place your image file** in this directory (`public/images/`)
2. **Rename your image** to `traffic-map-bg.jpg` (or update the filename in the code)
3. **Supported formats**: JPG, PNG, WEBP, SVG

## 🖼️ Recommended Image Specifications

- **Resolution**: 1920x1080 or higher for best quality
- **Aspect Ratio**: 16:9 or similar landscape format
- **File Size**: Under 2MB for optimal loading
- **Content**: Aerial city view, street map, or traffic-related imagery

## 🔧 Changing the Image Path

To use a different filename or add multiple images, update the path in:
`src/components/smart-traffic/SmartTrafficSystem.tsx`

```typescript
backgroundImage="/images/your-image-name.jpg"
```

## 📝 Current Configuration

- **Current Path**: `/images/traffic-map-bg.jpg`
- **Overlay**: Enabled (40% opacity)
- **Fallback**: Gradient background if image not found