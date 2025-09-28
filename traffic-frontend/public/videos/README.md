# Mini Videos for Traffic Intersections

## 📁 Video Directory Structure

Place your traffic intersection videos in this directory with the following naming convention:

```
public/videos/
├── rajmahal-square.mp4
├── kalpana-square.mp4
├── shastri-nagar.mp4
├── acharya-vihar.mp4
├── maharishi-college.mp4
└── video-placeholder.jpg (optional fallback image)
```

## 🎥 Video Requirements

### **File Formats Supported:**
- **MP4** (H.264) - Recommended for best compatibility
- **WebM** - Modern browsers
- **OGV** - Fallback support

### **Recommended Specifications:**
- **Resolution**: 720p (1280x720) or 480p (854x480)
- **Frame Rate**: 25-30 FPS
- **Duration**: 30-60 seconds (looping)
- **File Size**: Under 10MB per video for optimal loading
- **Aspect Ratio**: 16:9 (landscape)

### **Content Guidelines:**
- **Real Traffic Footage**: Actual intersection recordings
- **Clear View**: Good visibility of traffic flow
- **Stable Recording**: Minimal camera shake
- **Daytime Preferred**: Better visibility for analysis

## 🔧 Configuration

### **Adding New Videos:**

1. **Place video file** in this directory
2. **Update the CrossPath data** in `src/hooks/useSmartTrafficSystem.ts`:

```typescript
{
  id: 6,
  name: 'New Intersection',
  x: 50,
  y: 60,
  congestion: 'Medium',
  vehicles: 25,
  videoUrl: '/videos/new-intersection.mp4',  // ← Add your video
  liveStreamUrl: 'http://127.0.0.1:5000/live_feed_6', // Optional live stream
  isVideoEnabled: true
}
```

### **Live Stream Integration:**

For real-time feeds, configure the `liveStreamUrl` to point to your streaming server:
- Local: `http://127.0.0.1:5000/live_feed_1`
- Network: `http://your-server-ip:port/stream`
- RTSP: `rtsp://your-camera-ip/stream`

## 🌟 Features

✅ **Auto-looping** - Videos repeat automatically  
✅ **Play/Pause Controls** - User interaction  
✅ **Volume Control** - Muted by default  
✅ **Live Stream Priority** - Live feeds override recorded videos  
✅ **Fallback Support** - Graceful handling when videos unavailable  
✅ **Mobile Responsive** - Works on all devices  

## 📱 Live Stream Fallback

If live streams are not available, the system will:
1. Use recorded video (`videoUrl`)
2. Show placeholder with intersection info
3. Display "Video feed not available" message

## 🛠️ Troubleshooting

### Video Not Playing:
- Check file path and name spelling
- Ensure video format is supported
- Verify file permissions
- Check browser console for errors

### Large File Sizes:
- Compress videos using tools like FFmpeg
- Consider lower resolution (480p)
- Optimize bitrate for web delivery

### Live Stream Issues:
- Verify stream URL accessibility
- Check CORS settings for external streams
- Ensure streaming server is running

## 💡 Tips

- **Test locally** before deploying
- **Use consistent naming** for easy management
- **Keep backup copies** of original footage
- **Monitor file sizes** for performance
- **Consider CDN** for production deployment