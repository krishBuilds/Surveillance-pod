# InternVideo2.5 Web Interface

A simple web application for interacting with the InternVideo2.5 video analysis system.

## Features

- **Interactive Web UI**: Modern, responsive interface for video analysis
- **Custom Prompts**: Ask specific questions about video content
- **Frame Control**: Adjust number of frames (1-160) for analysis
- **FPS Sampling**: Control temporal sampling rate (0.1-5.0 FPS)
- **Time Bounds**: Limit analysis to specific video segments
- **Real-time Results**: Live display of AI responses and performance metrics
- **Singleton Architecture**: Uses the same optimized model instance as main.py

## Quick Start

1. **Start the web server:**
   ```bash
   cd /workspace/surveillance/webapp
   python app.py
   ```

2. **Access the interface:**
   Open your browser to: `http://localhost:8088`

3. **Configure and analyze:**
   - Enter your analysis prompt
   - Set desired frame count (max 160)
   - Optionally set FPS sampling or time bounds
   - Click "Start Analysis"

## Configuration Options

### Frame Count (1-160)
- **Low (16-48)**: Fast processing, basic analysis
- **Medium (60-120)**: Balanced performance and detail
- **High (140-160)**: Maximum detail, longer processing time

### FPS Sampling (Optional)
- **0.5 FPS**: Sparse temporal sampling for overview
- **1.0 FPS**: Dense sampling for detailed analysis
- **2.0+ FPS**: Very dense sampling (requires lower frame counts)

### Time Bounds (Optional)
- Limit analysis to first N seconds of video
- Useful for focusing on specific segments
- Combines with FPS sampling for precise control

## Example Prompts

### Story Analysis
```
Analyze the story progression, character development, and narrative structure in this video.
```

### Character Focus
```
Focus on the main character's personality traits, emotions, and interactions with other characters.
```

### Visual Analysis
```
Describe the visual composition, animation techniques, and artistic elements used in this video.
```

### Action Tracking
```
Track all actions and movements in the video, focusing on cause-and-effect relationships.
```

## Performance Expectations

### Processing Times (Approximate)
- **16 frames**: 30-45 seconds
- **60 frames**: 45-60 seconds  
- **120 frames**: 60-90 seconds
- **160 frames**: 90-120 seconds

### Memory Usage
- Uses the same singleton model instance
- No additional GPU memory overhead
- Automatic cleanup after each request

## Technical Details

- **Backend**: Flask web server
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Model Integration**: Subprocess calls to main.py
- **Error Handling**: Comprehensive error reporting
- **Timeout**: 10-minute maximum per request
- **Status Monitoring**: Real-time system status display

## API Endpoints

### GET /
Main interface page

### POST /analyze
Process video analysis request
```json
{
  "prompt": "Your analysis question",
  "num_frames": 120,
  "fps_sampling": 1.0,
  "time_bound": 160
}
```

### GET /status
System status and video information
```json
{
  "status": "online",
  "video_available": true,
  "video_path": "inputs/videos/street_scene.mp4",
  "video_info": {
    "fps": 24.0,
    "frames": 14315,
    "duration": "596.46s"
  },
  "max_frames": 160
}
```

## Troubleshooting

### Common Issues

**Video not found**
- Ensure `inputs/videos/street_scene.mp4` exists in parent directory
- Check file permissions

**Analysis timeout**
- Reduce frame count for faster processing
- Use FPS sampling to limit temporal density
- Set time bounds for shorter segments

**Memory errors**
- Stick to maximum 160 frames
- Avoid running multiple analyses simultaneously
- Restart webapp if needed

### Performance Tips

- Use 120 frames for optimal balance of detail and speed
- Set FPS sampling to 1.0 for consistent temporal coverage
- Use time bounds to focus on interesting segments
- Monitor system status before starting long analyses

## Integration

The web app integrates seamlessly with the existing InternVideo2.5 system:
- Uses the same singleton ModelManager
- Leverages all optimizations (8-bit quantization, etc.)
- Maintains the same performance characteristics
- No additional model loading overhead