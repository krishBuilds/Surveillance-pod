// InternVideo2.5 Surveillance - Main JavaScript

// Add the formatDuration function to the global scope for use in event handlers
function formatDuration(milliseconds) {
    if (milliseconds < 1000) return `${milliseconds}ms`;
    const seconds = milliseconds / 1000;
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Add the showNotification function to the global scope
function showNotification(message, type = 'info', duration = 5000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '10000';
    notification.style.padding = '12px 20px';
    notification.style.borderRadius = '8px';
    notification.style.maxWidth = '300px';
    notification.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
    
    // Set background color based on type
    const colors = {
        info: '#3498db',
        success: '#27ae60',
        error: '#e74c3c',
        warning: '#f39c12'
    };
    notification.style.backgroundColor = colors[type] || colors.info;
    notification.style.color = 'white';
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, duration);
}

// Timeline functions
let timeline = null;
let currentVideoData = null;
let currentCaptionData = null;

// Setup timeline video player
function setupTimelineVideoPlayer() {
    const timelineVideoPlayer = document.getElementById('timeline-video-player');
    const videoPlayer = document.getElementById('video-player');

    if (timelineVideoPlayer && videoPlayer && videoPlayer.src) {
        // Copy the source from main video player to timeline video player
        timelineVideoPlayer.src = videoPlayer.src;
        timelineVideoPlayer.load();
    }
}

async function showTimeline() {
    if (!currentCaptionData || currentCaptionData.length === 0) {
        showNotification('No caption data available for timeline generation. Please generate captions first.', 'warning');
        return;
    }

    // Hide chunks view and show timeline view
    const captionResults = document.getElementById('caption-results');
    const timelineView = document.getElementById('timeline-view');

    if (captionResults) captionResults.classList.add('hidden');
    if (timelineView) timelineView.classList.remove('hidden');

    // Initialize timeline if not already created
    if (!timeline) {
        initializeTimeline();
    }

    // Show loading state
    showNotification('Generating timeline...', 'info');

    try {
        // Try API-based timeline generation first (enhanced)
        try {
            const apiEvents = await generateAPITimeline();
            if (apiEvents && apiEvents.length > 0) {
                showNotification(`Enhanced timeline generated with ${apiEvents.length} events`, 'success');
                return;
            }
        } catch (apiError) {
            console.warn('API timeline generation failed, falling back to direct UI:', apiError);
        }

        // Fallback to simple timeline generation
        const events = generateSimpleTimeline();
        if (events && events.length > 0) {
            showNotification(`Timeline generated with ${events.length} events`, 'success');
        } else {
            showNotification('No events could be generated from caption data', 'warning');
        }

        // Ensure timeline container is visible
        const container = document.getElementById('video-timeline-container');
        if (container) {
            container.style.display = 'block';
        }
    } catch (error) {
        console.error('Timeline generation failed:', error);
        showNotification('Failed to generate timeline', 'error');
    }

    // Set up video player integration if available
    setupVideoPlayerIntegration();
}

async function generateEnhancedTimeline() {
    if (!currentCaptionData || !currentVideoData) {
        throw new Error('No caption or video data available');
    }

    showNotification('Generating enhanced timeline with AI...', 'info');

    // Prepare data for GPT-4o mini processing
    const captions = currentCaptionData.map(chunk => ({
        chunk_id: chunk.chunk_id || 1,
        start_time: chunk.start_time || 0,
        end_time: chunk.end_time || (chunk.start_time || 0) + 30,
        caption: chunk.caption || ''
    }));

    const videoInfo = {
        filename: currentVideoData.filename || 'unknown',
        duration: currentVideoData.duration || 0,
        fps_sampling: currentVideoData.fps_used || 3.0,
        processing_mode: currentVideoData.processing_mode || 'temporal_analysis'
    };

    const response = await fetch('/generate-timeline', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            video_file: currentVideoData.filename,
            captions: captions,
            video_info: videoInfo,
            total_duration: currentVideoData.duration || 0
        })
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Timeline generation failed');
    }

    const result = await response.json();

    if (result.success && result.timeline_data) {
        // Use the structured timeline data from GPT-4o mini
        const videoDuration = currentVideoData ? currentVideoData.duration || 0 : 0;
        timeline.setEvents(result.timeline_data, videoDuration);
        updateTimelineStats(result.timeline_data.events || [], videoDuration);

        // Store key frames and thumbnail suggestions for later use
        if (result.key_frames) {
            currentVideoData.key_frames = result.key_frames;
        }
        if (result.thumbnail_suggestions) {
            currentVideoData.thumbnail_suggestions = result.thumbnail_suggestions;
        }

        // Automatically extract thumbnails for significant events
        if (currentVideoData.filename && result.total_events > 0) {
            setTimeout(() => {
                timeline.extractThumbnailsForEvents(currentVideoData.filename);
            }, 1000); // Delay to allow timeline to render first
        }

        showNotification(`Generated ${result.total_events} timeline events with AI`, 'success');
    } else {
        throw new Error('Invalid timeline data received');
    }
}

function parseTimelineFromCaptions(captionData) {
    // Basic fallback timeline parsing from caption text
    const events = [];

    if (!captionData || !Array.isArray(captionData)) {
        return events;
    }

    captionData.forEach(chunk => {
        if (chunk.caption) {
            // Extract timestamp patterns from caption text
            const timestampRegex = /(At|Around|From)\s+(\d+(?:\.\d+)?)s(?:(?:\s+to\s+|\s*-\s*)(\d+(?:\.\d+)?)s?)?/gi;
            let match;

            while ((match = timestampRegex.exec(chunk.caption)) !== null) {
                const startTimestamp = parseFloat(match[2]);
                const endTimestamp = match[3] ? parseFloat(match[3]) : startTimestamp + 2;

                events.push({
                    timestamp: startTimestamp,
                    duration: endTimestamp - startTimestamp,
                    title: `Event at ${startTimestamp.toFixed(1)}s`,
                    description: chunk.caption.substring(0, 100) + '...',
                    category: 'action',
                    confidence: 0.7,
                    visualSignificance: 5
                });
            }

            // If no timestamps found, create a general event for the chunk
            if (events.length === 0) {
                events.push({
                    timestamp: chunk.start_time || 0,
                    duration: (chunk.end_time || (chunk.start_time || 0) + 30) - (chunk.start_time || 0),
                    title: `Activity in chunk ${chunk.chunk_id || 1}`,
                    description: chunk.caption.substring(0, 100) + '...',
                    category: 'action',
                    confidence: 0.6,
                    visualSignificance: 4
                });
            }
        }
    });

    return events.sort((a, b) => a.timestamp - b.timestamp);
}

function toggleTimelineView() {
    const timelineView = document.getElementById('timeline-view');
    const captionResults = document.getElementById('caption-results');

    if (timelineView.classList.contains('hidden')) {
        showTimeline();
    } else {
        timelineView.classList.add('hidden');
        captionResults.classList.remove('hidden');
    }
}

function initializeTimeline() {
    const container = document.getElementById('video-timeline-container');
    if (!container) {
        console.error('Timeline container not found');
        return;
    }

    // VideoTimeline class will be initialized when needed
    if (typeof VideoTimeline === 'undefined') {
        // Will use fallback UI method when timeline is generated
        return;
    }

    try {
        timeline = new VideoTimeline('video-timeline-container', {
            theme: 'light',
            animate: true,
            lineColor: '#3498db',
            dotColor: '#3498db',
            showThumbnails: true,
            thumbnailSize: 60,
            dateFormat: 'ss.SS'
        });

        // Listen for timeline seek events
        container.addEventListener('timelineSeek', (event) => {
            const timestamp = event.detail.timestamp;
            seekVideoToTimestamp(timestamp);
        });

        // Listen for timeline play events
        container.addEventListener('timelinePlay', (event) => {
            const timestamp = event.detail.timestamp;
            playVideoFromTimestamp(timestamp);
        });

        console.log('VideoTimeline initialized successfully');
    } catch (error) {
        console.error('Failed to initialize VideoTimeline:', error);
        timeline = null; // Reset to force fallback
    }
}

function updateTimelineStats(events, videoDuration) {
    const eventCount = document.getElementById('timeline-event-count');
    const durationBadge = document.getElementById('timeline-duration');

    if (eventCount) {
        eventCount.textContent = `${events.length} events`;
    }

    if (durationBadge) {
        const totalDuration = events.reduce((sum, event) => sum + event.duration, 0);
        durationBadge.textContent = `${totalDuration.toFixed(1)}s total`;
    }

    // Update timeline time display
    const timelineTime = document.getElementById('timeline-time');
    if (timelineTime) {
        timelineTime.textContent = `0.0s / ${videoDuration.toFixed(1)}s`;
    }
}

function setupVideoPlayerIntegration() {
    const videoPlayer = document.getElementById('video-player');
    if (!videoPlayer) return;

    // Update timeline progress as video plays
    videoPlayer.addEventListener('timeupdate', () => {
        if (timeline) {
            timeline.updateProgress(videoPlayer.currentTime);
            updateTimelineProgress(videoPlayer.currentTime, videoPlayer.duration);
        }
    });

    // Set video duration when metadata loads
    videoPlayer.addEventListener('loadedmetadata', () => {
        if (timeline) {
            timeline.setVideoDuration(videoPlayer.duration);
            updateTimelineStats(timeline.events, videoPlayer.duration);
        }
    });
}

function updateTimelineProgress(currentTime, duration) {
    const progressBar = document.getElementById('timeline-progress');
    const timelineTime = document.getElementById('timeline-time');

    if (progressBar && duration > 0) {
        const progress = (currentTime / duration) * 100;
        progressBar.style.width = `${progress}%`;
    }

    if (timelineTime && duration > 0) {
        timelineTime.textContent = `${currentTime.toFixed(1)}s / ${duration.toFixed(1)}s`;
    }
}

// Simplified timeline generation that always works
function generateSimpleTimeline() {
    if (!currentCaptionData || currentCaptionData.length === 0) {
        showNotification('No caption data available', 'warning');
        return;
    }

    // Create timeline events directly from chunks with exact timestamps
    const events = currentCaptionData.map(chunk => {
        const startTime = parseFloat(chunk.start_time) || 0;
        const endTime = parseFloat(chunk.end_time) || (startTime + 30);
        const duration = endTime - startTime;

        // Extract a meaningful title from the caption (first sentence or first few words)
        const caption = chunk.caption || 'No caption available';
        const title = caption.length > 50 ? caption.substring(0, 50) + '...' : caption;

        return {
            timestamp: startTime,
            duration: duration,
            title: title,
            description: caption,
            type: 'action',
            confidence: 0.7
        };
    });

    const videoDuration = currentVideoData ? parseFloat(currentVideoData.duration) || 0 : 0;

    // Always use VideoTimeline component for caption page
    if (typeof VideoTimeline === 'undefined') {
        throw new Error('VideoTimeline component not available');
    }

    const container = document.getElementById('video-timeline-container');
    if (!timeline) {
        timeline = new VideoTimeline('video-timeline-container', {
            theme: 'light',
            animate: true,
            lineColor: '#4f46e5',
            dotColor: '#7c3aed',
            showThumbnails: true,
            thumbnailSize: 60,
            dateFormat: 'ss.SS'
        });

        // Add video seek functionality to the timeline
        if (timeline.seekToVideo) {
            timeline.seekToVideo = (timestamp) => {
                seekVideoToTimestamp(timestamp);
            };
        }
    }

    timeline.setEvents(events, videoDuration);

    // Auto-extract frames for the first few events
    setTimeout(async () => {
        if (currentVideoData?.filename) {
            try {
                await extractFramesForTimeline(events);
                console.log('Frames extracted successfully for timeline');
            } catch (error) {
                console.warn('Auto frame extraction failed:', error);
            }
        }
    }, 1000);

    showNotification(`Timeline generated with ${events.length} events`, 'success');

    return events;
}

// API-based timeline generation for enhanced functionality
async function generateAPITimeline() {
    if (!currentCaptionData || !currentVideoData) {
        throw new Error('No caption or video data available');
    }

    showNotification('Generating enhanced timeline via API...', 'info');

    // Prepare data for API
    const captions = currentCaptionData.map(chunk => ({
        chunk_id: chunk.chunk_id || 1,
        start_time: chunk.start_time || 0,
        end_time: chunk.end_time || (chunk.start_time || 0) + 30,
        caption: chunk.caption || ''
    }));

    const videoInfo = {
        filename: currentVideoData.filename || 'unknown',
        duration: currentVideoData.duration || 0,
        fps_sampling: currentVideoData.fps_used || 3.0,
        processing_mode: currentVideoData.processing_mode || 'temporal_analysis'
    };

    try {
        const response = await fetch('/generate-timeline', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                video_file: currentVideoData.filename,
                captions: captions,
                video_info: videoInfo,
                total_duration: currentVideoData.duration || 0
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Timeline generation failed');
        }

        const result = await response.json();

        console.log('API Timeline Response:', result);

        if (result.success && result.timeline_data && result.timeline_data.events) {
            // Transform API events to timeline format - preserve exact GPT timestamps
            const timelineEvents = result.timeline_data.events.map(event => ({
                timestamp: parseFloat(event.start_time) || 0, // Use exact start_time from GPT
                duration: parseFloat(event.duration) || (parseFloat(event.end_time) || 0) - (parseFloat(event.start_time) || 0),
                title: event.title || 'Event',
                description: event.description || '',
                type: event.category || 'action', // Use category from GPT response
                confidence: parseFloat(event.confidence) || 0.8,
                thumbnail: event.thumbnail || null
            }));

            // Always use VideoTimeline component for caption page
            const container = document.getElementById('video-timeline-container');
            if (typeof VideoTimeline === 'undefined') {
                throw new Error('VideoTimeline component not available');
            }

            if (!timeline) {
                timeline = new VideoTimeline('video-timeline-container', {
                    theme: 'light',
                    animate: true,
                    lineColor: '#4f46e5',
                    dotColor: '#7c3aed',
                    showThumbnails: true,
                    thumbnailSize: 60,
                    dateFormat: 'ss.SS'
                });

                // Add video seek functionality to the timeline
                if (timeline.seekToVideo) {
                    timeline.seekToVideo = (timestamp) => {
                        seekVideoToTimestamp(timestamp);
                    };
                }
            }

            timeline.setEvents(timelineEvents, currentVideoData.duration || 0);

            // Auto-extract frames for the first few events
            setTimeout(async () => {
                if (currentVideoData?.filename) {
                    try {
                        await extractFramesForTimeline(timelineEvents);
                        console.log('Frames extracted successfully for API timeline');
                    } catch (error) {
                        console.warn('Auto frame extraction failed:', error);
                    }
                }
            }, 1000);

            showNotification(`Timeline generated with ${timelineEvents.length} events`, 'success');
            return timelineEvents;
        }

        throw new Error('Invalid timeline response format');
    } catch (error) {
        console.warn('API timeline generation failed:', error);
        throw error; // Re-throw to trigger fallback
    }
}

// Clean, minimal timeline UI for caption page
function createDirectTimelineUI(events, videoDuration) {
    const container = document.getElementById('video-timeline-container');
    if (!container) {
        console.error('Timeline container not found');
        return;
    }

    // Clear container
    container.innerHTML = '';

    // Create minimal timeline structure
    const timelineHTML = `
        <div class="minimal-timeline" style="max-width: 400px; margin: 0 auto; padding: 10px;">
            <!-- Compact Header -->
            <div style="text-align: center; margin-bottom: 15px;">
                <div style="font-size: 14px; font-weight: 600; color: #374151;">
                    ${events.length} events • ${videoDuration.toFixed(1)}s
                </div>
            </div>

            <!-- Vertical Timeline -->
            <div style="position: relative; padding-left: 25px;">
                <!-- Timeline Line -->
                <div style="position: absolute; left: 8px; top: 0; bottom: 0; width: 1px; background: #d1d5db;"></div>

                ${events.map((event, index) => {
                    const middleFrameTime = event.timestamp + (event.duration / 2);
                    return `
                        <div class="timeline-event" style="position: relative; margin-bottom: 12px;" data-timestamp="${event.timestamp}">
                            <!-- Simple Timeline Node -->
                            <div style="
                                position: absolute;
                                left: -20px;
                                top: 4px;
                                width: 8px;
                                height: 8px;
                                background: #4f46e5;
                                border-radius: 50%;
                                border: 1px solid white;
                                cursor: pointer;
                            " onclick="seekVideoToTimestamp(${event.timestamp})"></div>

                            <!-- Minimal Event Card -->
                            <div style="
                                background: white;
                                border: 1px solid #e5e7eb;
                                border-radius: 6px;
                                padding: 8px 10px;
                                margin-left: 10px;
                                cursor: pointer;
                                transition: all 0.2s ease;
                                font-size: 12px;
                            " onclick="seekVideoToTimestamp(${event.timestamp})">
                                <!-- Time and Description -->
                                <div style="display: flex; align-items: flex-start; gap: 8px;">
                                    <span style="color: #6b7280; font-weight: 500; white-space: nowrap;">
                                        ${event.timestamp.toFixed(1)}s
                                    </span>
                                    <span style="color: #374151; flex: 1; line-height: 1.3;">
                                        ${event.description.length > 80 ? event.description.substring(0, 80) + '...' : event.description}
                                    </span>
                                </div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
    `;

    container.innerHTML = timelineHTML;

    // Add minimal hover effect
    container.querySelectorAll('.timeline-event > div:last-child').forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.backgroundColor = '#f9fafb';
            card.style.borderColor = '#4f46e5';
        });

        card.addEventListener('mouseleave', () => {
            card.style.backgroundColor = 'white';
            card.style.borderColor = '#e5e7eb';
        });
    });

    // Auto-extract frames for timeline thumbnails
    setTimeout(() => {
        extractFramesForTimeline(events);
    }, 500);
}


function seekVideoToTimestamp(timestamp) {
    const videoPlayer = document.getElementById('video-player');
    if (videoPlayer) {
        videoPlayer.currentTime = timestamp;
        videoPlayer.play();
    } else {
        showNotification(`Seeking to ${timestamp.toFixed(1)}s`, 'info');
    }
}

function playVideoFromTimestamp(timestamp) {
    // Try to find and use any available video player
    let videoPlayer = document.getElementById('video-player');

    // If main video player is not available or hidden, try the preview player
    if (!videoPlayer || videoPlayer.offsetParent === null) {
        videoPlayer = document.getElementById('caption-preview-player');
    }

    if (videoPlayer) {
        // Ensure the video has a source loaded
        if (!videoPlayer.src && currentVideoData?.filename) {
            // Load the video source if not already loaded
            videoPlayer.src = `/serve-video/${currentVideoData.filename}`;
        }

        // Show the video preview section if it's hidden
        const videoPreview = document.getElementById('video-preview');
        if (videoPreview && videoPreview.classList.contains('hidden')) {
            videoPreview.classList.remove('hidden');
            // Hide timeline view temporarily
            const timelineView = document.getElementById('timeline-view');
            if (timelineView) {
                timelineView.classList.add('hidden');
            }
            // Show caption results
            const captionResults = document.getElementById('caption-results');
            if (captionResults) {
                captionResults.classList.remove('hidden');
            }
        }

        videoPlayer.currentTime = timestamp;
        videoPlayer.play()
            .then(() => {
                console.log(`▶️ Playing video from ${timestamp.toFixed(1)}s`);
                showNotification(`Playing from ${timestamp.toFixed(1)}s`, 'success', 2000);
            })
            .catch((error) => {
                console.warn('Video play failed:', error);
                showNotification(`Seeked to ${timestamp.toFixed(1)}s`, 'info', 2000);
            });
    } else {
        showNotification(`No video player available for ${timestamp.toFixed(1)}s`, 'warning');
    }
}

function formatTime(seconds) {
    if (!seconds || seconds < 0) return '0.0s';
    const secs = Math.floor(seconds);
    const ms = Math.round((seconds - secs) * 100);
    return `${secs}.${ms.toString().padStart(2, '0')}s`;
}

async function extractFramesForTimeline(events) {
    if (!events || !currentVideoData?.filename) {
        return;
    }

    console.log('🖼️ Extracting frames for timeline events...');

    // Extract frames for each timeline event
    for (let i = 0; i < Math.min(events.length, 10); i++) { // Limit to first 10 events
        const event = events[i];
        const frameTime = event.timestamp + (event.duration / 2); // Middle frame

        try {
            const frameUrl = await extractFrameAtTime(currentVideoData.filename, frameTime);

            // Find the timeline event and update it with the frame
            const timelineEvent = document.querySelector(`[data-timestamp="${event.timestamp}"]`);
            if (timelineEvent) {
                // Find the frame container and update it
                const frameContainer = timelineEvent.querySelector('.timeline-frame-container.placeholder');

                if (frameContainer) {
                    // Replace placeholder with actual frame
                    frameContainer.classList.remove('placeholder');
                    frameContainer.innerHTML = `
                        <img src="${frameUrl}"
                             alt="Frame at ${formatTime(event.timestamp)}"
                             class="timeline-frame-image"
                             onclick="timeline.seekToVideo(${event.timestamp})"
                             title="Click to seek to ${formatTime(event.timestamp)}" />
                        <div class="frame-time-overlay">${formatTime(event.timestamp)}</div>
                        <div class="frame-play-overlay" onclick="event.stopPropagation(); timeline.playFromTimestamp(${event.timestamp})" title="Play from this moment">
                            <i class="bi bi-play-fill"></i>
                        </div>
                    `;

                    console.log(`✅ Frame extracted for event at ${frameTime.toFixed(1)}s`);
                }
            }
        } catch (error) {
            console.warn(`❌ Failed to extract frame for event at ${frameTime}s:`, error);
        }
    }
}

async function extractFrameAtTime(videoFilename, timestamp) {
    try {
        const response = await fetch('/extract-frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                video_file: videoFilename,
                timestamp: timestamp
            })
        });

        if (!response.ok) {
            throw new Error(`Frame extraction failed: ${response.status}`);
        }

        const data = await response.json();
        if (data.success && data.image_data) {
            return data.image_data;
        } else {
            throw new Error(data.error || 'No frame data returned');
        }
    } catch (error) {
        console.error('Frame extraction error:', error);
        throw error;
    }
}

async function extractTimelineThumbnails() {
    if (!timeline || !timeline.events || !currentVideoData?.filename) {
        showNotification('No timeline data or video file available', 'warning');
        return;
    }

    await timeline.extractThumbnailsForEvents(currentVideoData.filename);
}

async function extractAllKeyframes() {
    if (!timeline || !timeline.events || !currentVideoData?.filename) {
        showNotification('No timeline data or video file available', 'warning');
        return;
    }

    showNotification('Extracting all key frames...', 'info');

    try {
        const keyFrames = await timeline.extractKeyFramesForEvents(timeline.events, currentVideoData.filename);

        if (keyFrames.length > 0) {
            // Store key frames for later use
            if (!currentVideoData.extracted_keyframes) {
                currentVideoData.extracted_keyframes = [];
            }
            currentVideoData.extracted_keyframes.push(...keyFrames);

            showNotification(`Extracted ${keyFrames.length} key frames successfully`, 'success');

            // Optionally create a downloadable gallery
            if (confirm('Would you like to download the key frames gallery?')) {
                downloadKeyFramesGallery(keyFrames);
            }
        } else {
            showNotification('No key frames found to extract', 'info');
        }
    } catch (error) {
        console.error('Error extracting key frames:', error);
        showNotification('Failed to extract key frames', 'error');
    }
}

function downloadKeyFramesGallery(keyFrames) {
    if (!keyFrames.length) return;

    // Create HTML gallery
    let html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Key Frames Gallery - ${currentVideoData?.filename || 'Video'}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; }
        .frame-card { background: white; border-radius: 8px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .frame-image { width: 100%; height: 120px; object-fit: cover; border-radius: 4px; }
        .frame-info { margin-top: 8px; font-size: 0.9rem; }
        .timestamp { font-weight: bold; color: #007bff; }
    </style>
</head>
<body>
    <h1>Key Frames Gallery</h1>
    <p><strong>Video:</strong> ${currentVideoData?.filename || 'Unknown'}</p>
    <p><strong>Total Frames:</strong> ${keyFrames.length}</p>
    <div class="gallery">
`;

    keyFrames.forEach((frame, index) => {
        html += `
        <div class="frame-card">
            <img src="${frame.image_data}" alt="Frame at ${frame.timestamp}s" class="frame-image">
            <div class="frame-info">
                <div class="timestamp">${formatTime(frame.timestamp)}</div>
                <div>Frame ${index + 1}</div>
            </div>
        </div>
    `;
    });

    html += `
    </div>
</body>
</html>`;

    // Download the HTML file
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `keyframes_gallery_${currentVideoData?.filename || 'video'}_${Date.now()}.html`;
    link.click();
    URL.revokeObjectURL(url);
}

function exportTimeline() {
    if (!timeline || !timeline.events) {
        showNotification('No timeline data to export', 'warning');
        return;
    }

    const timelineData = {
        video_file: currentVideoData?.filename || 'unknown',
        duration: currentVideoData?.duration || 0,
        events: timeline.events,
        totalEvents: timeline.events.length,
        totalDuration: timeline.events.reduce((sum, event) => sum + event.duration, 0),
        generated_at: new Date().toISOString(),
        ai_enhanced: !!currentVideoData?.key_frames,
        key_frames_count: currentVideoData?.key_frames?.length || 0,
        thumbnail_suggestions: currentVideoData?.thumbnail_suggestions || []
    };

    const dataStr = JSON.stringify(timelineData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });

    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `timeline_${currentVideoData?.filename || 'video'}_${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    showNotification('Timeline exported successfully', 'success');
}

class SurveillanceApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupEventListeners();
        this.loadInitialPage();
    }

    setupNavigation() {
        // Handle navigation clicks
        document.addEventListener('click', (e) => {
            if (e.target.matches('.nav-link')) {
                e.preventDefault();
                const page = e.target.dataset.page;
                this.navigateToPage(page);
                this.setActiveNavItem(e.target);
            }
        });
    }

    setupEventListeners() {
        // Mobile sidebar toggle
        const sidebarToggle = document.querySelector('.sidebar-toggle');
        const sidebar = document.querySelector('.sidebar');
        
        if (sidebarToggle && sidebar) {
            sidebarToggle.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });
        }
    }

    loadInitialPage() {
        // Load the default page (caption interface)
        const defaultNavItem = document.querySelector('.nav-link[data-page="caption"]');
        if (defaultNavItem) {
            this.navigateToPage('caption');
            this.setActiveNavItem(defaultNavItem);
        }
    }

    navigateToPage(page) {
        // Hide all page content with smooth transition
        document.querySelectorAll('.page-section').forEach(content => {
            content.classList.add('hidden');
        });

        // Show selected page with smooth transition
        const targetPage = document.querySelector(`#${page}-page`);
        if (targetPage) {
            // Small delay to ensure smooth transition
            setTimeout(() => {
                targetPage.classList.remove('hidden');
            }, 50);
        }

        // Load page-specific content if needed
        switch (page) {
            case 'caption':
                this.loadCaptionPage();
                break;
            case 'analysis':
                this.loadAnalysisPage();
                break;
            case 'timeline':
                this.loadTimelinePage();
                break;
        }
    }

    setActiveNavItem(activeItem) {
        // Remove active class from all nav items
        document.querySelectorAll('.nav-link').forEach(item => {
            item.classList.remove('active');
        });

        // Add active class to clicked item
        activeItem.classList.add('active');
    }

    
    loadCaptionPage() {
        // Initialize caption generation functionality
        if (window.CaptionGenerator) {
            window.CaptionGenerator.init();
        }
        
        // Setup video preview functionality
        setTimeout(() => {
            if (window.setupVideoPreview) {
                window.setupVideoPreview();
            }
        }, 100);
    }

    loadAnalysisPage() {
        // Initialize analysis functionality
        console.log('Analysis page loaded');
        this.setupAnalysisPage();

        // Setup video preview functionality
        setTimeout(() => {
            if (window.setupVideoPreview) {
                window.setupVideoPreview();
            }
        }, 100);
    }

    loadTimelinePage() {
        // Initialize timeline functionality
        console.log('Timeline page loaded');
        this.setupTimelinePage();

        // Setup video preview functionality
        setTimeout(() => {
            if (window.setupVideoPreview) {
                window.setupVideoPreview();
            }
        }, 100);
    }

    async setupAnalysisPage() {
        // Load stored captions for video selection
        await this.loadStoredCaptionsForAnalysis();
        
        // Setup event listeners
        const videoSelector = document.getElementById('video-selector');
        const analyzeBtn = document.getElementById('analyze-chunks-btn');
        const timelineBtn = document.getElementById('show-timeline-btn');

        if (videoSelector) {
            videoSelector.addEventListener('change', async (e) => {
                const videoFile = e.target.value;
                const temporalBtn = document.getElementById('temporal-analysis-btn');

                console.log('🎯 Video selector changed to:', videoFile);

                if (videoFile) {
                    try {
                        console.log('📞 Calling displayVideoInfoForAnalysis...');
                        await this.displayVideoInfoForAnalysis(videoFile);
                        console.log('✅ displayVideoInfoForAnalysis completed');

                        console.log('📞 Calling loadChunkDataForAnalysis...');
                        await this.loadChunkDataForAnalysis(videoFile);
                        console.log('✅ loadChunkDataForAnalysis completed');

                        if (analyzeBtn) analyzeBtn.disabled = false;
                        if (temporalBtn) temporalBtn.disabled = false;
                        if (timelineBtn) timelineBtn.disabled = false;
                    } catch (error) {
                        console.error('❌ Error in video selector change handler:', error);
                    }
                } else {
                    this.hideVideoInfoDisplay();
                    if (analyzeBtn) analyzeBtn.disabled = true;
                    if (temporalBtn) temporalBtn.disabled = true;
                    if (timelineBtn) timelineBtn.disabled = true;
                }
            });
        }

        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', async () => {
                const videoFile = videoSelector?.value;
                const query = document.getElementById('chunk-query')?.value?.trim();
                console.log('Analyze button clicked:', { videoFile, query });
                if (videoFile && query) {
                    await this.findRelevantChunk(videoFile, query);
                } else {
                    console.error('Missing videoFile or query:', { videoFile, query });
                    showNotification('Please select a video and enter a query', 'error');
                }
            });
        }

        // Temporal analysis button
        const temporalBtn = document.getElementById('temporal-analysis-btn');
        if (temporalBtn) {
            temporalBtn.addEventListener('click', async () => {
                const videoFile = videoSelector?.value;
                const query = document.getElementById('chunk-query')?.value?.trim();
                const startTime = parseFloat(document.getElementById('temporal-start-time')?.value) || 0;
                const endTime = parseFloat(document.getElementById('temporal-end-time')?.value) || 0;

                if (videoFile && query && startTime >= 0 && endTime > startTime) {
                    await this.performTemporalAnalysis(videoFile, query, startTime, endTime);
                } else {
                    showNotification('❌ Please fill in query and valid time range', 'error', 3000);
                }
            });
        }

        // Timeline button
        if (timelineBtn) {
            timelineBtn.addEventListener('click', async () => {
                const videoFile = videoSelector?.value;
                if (videoFile) {
                    await this.showTimelineForAnalysis(videoFile);
                } else {
                    showNotification('Please select a video first', 'error');
                }
            });
        }
    }

    async setupTimelinePage() {
        // Load processed videos for timeline generation
        await this.loadProcessedVideosForTimeline();

        // Setup event listeners
        const videoSelector = document.getElementById('timeline-video-selector');
        const generateBtn = document.getElementById('generate-timeline-btn');
        const viewExistingBtn = document.getElementById('view-existing-timeline-btn');

        if (videoSelector) {
            videoSelector.addEventListener('change', async (e) => {
                const videoFile = e.target.value;
                if (videoFile) {
                    await this.loadTimelineVideoInfo(videoFile);
                    generateBtn.disabled = false;
                    viewExistingBtn.disabled = false;
                } else {
                    generateBtn.disabled = true;
                    viewExistingBtn.disabled = true;
                }
            });
        }

        if (generateBtn) {
            generateBtn.addEventListener('click', async () => {
                const videoFile = videoSelector?.value;
                if (videoFile) {
                    await this.generateTimeline(videoFile);
                }
            });
        }

        if (viewExistingBtn) {
            viewExistingBtn.addEventListener('click', async () => {
                const videoFile = videoSelector?.value;
                if (videoFile) {
                    await this.viewExistingTimeline(videoFile);
                }
            });
        }
    }

    async loadProcessedVideosForTimeline() {
        try {
            const response = await fetch('/api/videos/processed');
            const data = await response.json();

            if (data.success && data.videos) {
                const selector = document.getElementById('timeline-video-selector');
                if (selector) {
                    selector.innerHTML = '<option value="">Choose a processed video...</option>';
                    data.videos.forEach(video => {
                        const option = document.createElement('option');
                        option.value = video.filename;
                        option.textContent = `${video.filename} (${video.chunks} chunks, ${video.duration}s)`;
                        selector.appendChild(option);
                    });
                }
            }
        } catch (error) {
            console.error('Error loading processed videos for timeline:', error);
            showNotification('Failed to load processed videos', 'error');
        }
    }

    async loadTimelineVideoInfo(videoFile) {
        try {
            const response = await fetch(`/api/video-caption/${encodeURIComponent(videoFile)}`);
            const data = await response.json();

            if (data.success && data.caption) {
                const infoDiv = document.getElementById('timeline-video-info');
                if (infoDiv) {
                    const chunks = data.caption.chunks ? data.caption.chunks.length : 0;
                    const duration = data.caption.video_duration || 0;

                    document.getElementById('timeline-chunks-display').textContent = chunks;
                    document.getElementById('timeline-duration-display').textContent = `${duration.toFixed(1)}s`;

                    infoDiv.classList.remove('hidden');
                }
            }
        } catch (error) {
            console.error('Error loading timeline video info:', error);
        }
    }

    async generateTimeline(videoFile) {
        try {
            showNotification('Generating timeline...', 'info');

            // Show loading state
            const loadingDiv = document.getElementById('timeline-loading');
            const controlsDiv = document.getElementById('timeline-controls');
            const emptyDiv = document.getElementById('timeline-empty-state');

            if (loadingDiv) loadingDiv.classList.remove('hidden');
            if (controlsDiv) controlsDiv.classList.add('hidden');
            if (emptyDiv) emptyDiv.classList.add('hidden');

            // Get timeline options
            const enhanced = document.getElementById('enhanced-timeline')?.checked || false;
            const autoExtract = document.getElementById('auto-extract-thumbnails')?.checked || false;

            // Fetch caption data
            const response = await fetch(`/api/video-caption/${encodeURIComponent(videoFile)}`);
            const data = await response.json();

            if (data.success && data.caption) {
                // Initialize timeline component
                if (window.VideoTimeline) {
                    await window.VideoTimeline.generateTimeline(data.caption, {
                        enhanced: enhanced,
                        autoExtractThumbnails: autoExtract,
                        videoFile: videoFile
                    });

                    // Show timeline controls
                    if (loadingDiv) loadingDiv.classList.add('hidden');
                    if (controlsDiv) controlsDiv.classList.remove('hidden');

                    showNotification('Timeline generated successfully', 'success');
                } else {
                    showNotification('Timeline component not available', 'error');
                }
            } else {
                showNotification('No caption data found for this video', 'error');
            }
        } catch (error) {
            console.error('Error generating timeline:', error);
            showNotification(`Failed to generate timeline: ${error.message}`, 'error');
        }
    }

    async viewExistingTimeline(videoFile) {
        try {
            showNotification('Loading existing timeline...', 'info');

            // Show loading state
            const loadingDiv = document.getElementById('timeline-loading');
            const controlsDiv = document.getElementById('timeline-controls');
            const emptyDiv = document.getElementById('timeline-empty-state');

            if (loadingDiv) loadingDiv.classList.remove('hidden');
            if (controlsDiv) controlsDiv.classList.add('hidden');
            if (emptyDiv) emptyDiv.classList.add('hidden');

            // Fetch caption data
            const response = await fetch(`/api/video-caption/${encodeURIComponent(videoFile)}`);
            const data = await response.json();

            if (data.success && data.caption) {
                // Initialize timeline component with existing data
                if (window.VideoTimeline) {
                    await window.VideoTimeline.loadExistingTimeline(data.caption, {
                        videoFile: videoFile
                    });

                    // Show timeline controls
                    if (loadingDiv) loadingDiv.classList.add('hidden');
                    if (controlsDiv) controlsDiv.classList.remove('hidden');

                    showNotification('Timeline loaded successfully', 'success');
                } else {
                    showNotification('Timeline component not available', 'error');
                }
            } else {
                showNotification('No existing timeline found for this video', 'error');
            }
        } catch (error) {
            console.error('Error loading existing timeline:', error);
            showNotification(`Failed to load timeline: ${error.message}`, 'error');
        }
    }

    async loadStoredCaptionsForAnalysis() {
        try {
            const response = await fetch('/api/stored-captions');
            const result = await response.json();
            const videos = result.videos || [];
            
            const selector = document.getElementById('video-selector');
            if (selector) {
                selector.innerHTML = '<option value="">Choose a video with captions...</option>';
                videos.forEach(video => {
                    const option = document.createElement('option');
                    option.value = video.video_file;
                    const createdDate = new Date(video.created_at).toLocaleDateString();
                    option.textContent = `${video.video_file} (${createdDate})`;
                    selector.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Error loading stored captions:', error);
        }
    }

    async displayVideoInfoForAnalysis(videoFile) {
        try {
            const url = `/api/video-caption/${encodeURIComponent(videoFile)}`;
            console.log('📡 displayVideoInfoForAnalysis requesting:', url);
            const response = await fetch(url);
            const result = await response.json();
            const data = result.data || result; // Handle nested data structure
            
            console.log('📥 displayVideoInfoForAnalysis response:', result);
            console.log('📥 displayVideoInfoForAnalysis data:', data);
            console.log('📥 displayVideoInfoForAnalysis success check:', {
                hasData: !!data,
                success: result.success,
                successNotFalse: result.success !== false
            });

            if (data && result.success === true) {
                const infoDisplay = document.getElementById('video-info-display');
                if (infoDisplay) {
                    infoDisplay.style.display = 'block';
                    
                    // Update chunk size
                    const chunkSizeDisplay = document.getElementById('chunk-size-display');
                    if (chunkSizeDisplay) {
                        const chunkDuration = data.chunk_duration || 48;
                        chunkSizeDisplay.textContent = `${chunkDuration}s`;
                    }
                    
                    // Update total chunks
                    const totalChunksDisplay = document.getElementById('total-chunks-display');
                    if (totalChunksDisplay && data.chunks) {
                        totalChunksDisplay.textContent = data.chunks.length;
                    }
                    
                    // Update FPS
                    const fpsDisplay = document.getElementById('fps-display');
                    if (fpsDisplay) {
                        fpsDisplay.textContent = `${data.fps || 2.0} FPS`;
                    }
                    
                    // Update duration with proper formatting
                    const durationDisplay = document.getElementById('duration-display');
                    if (durationDisplay) {
                        const duration = data.video_duration || 0;
                        const minutes = Math.floor(duration / 60);
                        const seconds = Math.floor(duration % 60);
                        durationDisplay.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
                    }
                }
            }
        } catch (error) {
            console.error('Error displaying video info:', error);
            this.hideVideoInfoDisplay();
        }
    }

    hideVideoInfoDisplay() {
        const infoDisplay = document.getElementById('video-info-display');
        if (infoDisplay) {
            infoDisplay.style.display = 'none';
        }
    }

    async loadChunkDataForAnalysis(videoFile) {
        try {
            const url = `/api/video-caption/${encodeURIComponent(videoFile)}`;
            console.log('📥 loadChunkDataForAnalysis fetching:', url);
            console.log('📥 loadChunkDataForAnalysis videoFile:', videoFile);

            const response = await fetch(url);
            const result = await response.json();
            const data = result.data || result; // Handle nested data structure

            console.log('📥 loadChunkDataForAnalysis response:', result);
            console.log('📥 Raw API response structure:', {
                hasData: !!result.data,
                dataKeys: result.data ? Object.keys(result.data) : [],
                chunksInData: !!result.data?.chunks,
                chunksInRoot: !!result.chunks
            });
            console.log('📥 loadChunkDataForAnalysis chunks:', data.chunks ? data.chunks.length : 'none');

            const chunkDataTextarea = document.getElementById('chunk-data');
            const currentVideoFile = document.getElementById('current-video-file');

            if (chunkDataTextarea && data.chunks) {
                // Format chunks for display
                const formattedChunks = data.chunks.map(chunk => ({
                    chunk_id: chunk.chunk_id || 1,
                    start_time: chunk.start_time || 0,
                    end_time: chunk.end_time || 0,
                    caption: chunk.caption || ''
                }));
                chunkDataTextarea.value = JSON.stringify(formattedChunks, null, 2);
                console.log('✅ Chunks loaded into textarea:', formattedChunks.length);
            } else {
                console.warn('⚠️ No chunks found in response data');
            }

            // Store current video file for segment playback
            if (currentVideoFile) {
                currentVideoFile.value = videoFile;
                console.log('✅ Current video file set to:', videoFile);
            }
        } catch (error) {
            console.error('❌ Error loading chunk data:', error);
        }
    }

    async findRelevantChunk(videoFile, query) {
        const loadingDiv = document.getElementById('chunk-analysis-loading');
        const resultsDiv = document.getElementById('chunk-analysis-results');
        const placeholderDiv = document.getElementById('chunk-analysis-placeholder');
        
        // Show loading state with timing
        const analysisTimer = this.startTiming();
        if (loadingDiv) loadingDiv.style.display = 'block';
        if (resultsDiv) resultsDiv.style.display = 'none';
        if (placeholderDiv) placeholderDiv.style.display = 'none';
        
        // Show start notification
        showNotification('🔍 Starting chunk analysis...', 'info', 3000);
        
        try {
            // First, fetch the stored captions for this video
            const captionsUrl = `/api/video-caption/${encodeURIComponent(videoFile)}`;
            console.log('Fetching captions from:', captionsUrl);
            console.log('Video file:', videoFile);

            const captionsResponse = await fetch(captionsUrl);
            const captionsData = await captionsResponse.json();

            console.log('Captions response:', captionsData);

            console.log('🔍 Validating captions data structure:');
            console.log('   captionsData.success:', captionsData.success);
            console.log('   captionsData.data exists:', !!captionsData.data);
            console.log('   captionsData.data.chunks exists:', !!captionsData.data?.chunks);
            console.log('   captionsData.data.chunks length:', captionsData.data?.chunks?.length || 0);
            console.log('   captionsData.chunks exists (root):', !!captionsData.chunks);
            console.log('   captionsData.chunks length (root):', captionsData.chunks?.length || 0);

            // Handle both nested and flat response structures
            const data = captionsData.data || captionsData;
            const chunks = data.chunks || [];

            console.log('🔍 Detailed chunk analysis:');
            console.log('   Raw captionsData:', captionsData);
            console.log('   Extracted data:', data);
            console.log('   Chunks array:', chunks);
            console.log('   Chunks type:', typeof chunks);
            console.log('   Chunks isArray:', Array.isArray(chunks));
            console.log('   Chunks length:', chunks.length);
            if (chunks && chunks.length > 0) {
                console.log('   First chunk:', chunks[0]);
                console.log('   First chunk type:', typeof chunks[0]);
            }

            if (!captionsData.success || !chunks || chunks.length === 0) {
                console.error('❌ Invalid captions data structure:', {
                    success: captionsData.success,
                    hasData: !!captionsData.data,
                    hasChunks: !!chunks,
                    chunksLength: chunks.length,
                    fullResponse: captionsData
                });
                throw new Error(`No caption chunks found for this video. Debug: success=${captionsData.success}, hasChunks=${!!chunks}, chunksLength=${chunks.length}`);
            }
            console.log('✅ Found chunks:', chunks.length, 'chunks');
            console.log('📤 Sending to backend:', {
                query: query,
                chunksCount: chunks.length,
                firstChunkPreview: chunks[0]?.caption?.substring(0, 100) + '...'
            });
            
            const response = await fetch('/api/find-relevant-chunk', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    chunks: chunks
                })
            });
            
            const result = await response.json();
            const processingTime = analysisTimer.end();
            
            if (result.success && result.chunk_details) {
                // Hide loading and show results
                if (loadingDiv) loadingDiv.style.display = 'none';
                if (resultsDiv) resultsDiv.style.display = 'block';
                
                // Calculate and display duration properly
                const startTime = result.chunk_details.start_time || 0;
                const endTime = result.chunk_details.end_time || 0;
                const duration = endTime - startTime;
                
                // Update result displays with proper formatting
                const elements = {
                    'result-chunk-number': result.relevant_chunk_number || '-',
                    'result-start-time': this.formatTime(startTime),
                    'result-end-time': this.formatTime(endTime),
                    'result-duration': duration.toFixed(1),
                    'result-caption': result.chunk_details.caption || '-',
                    'result-analysis': result.analysis || 'No analysis available',
                    'result-confidence': result.confidence_score || '-',
                    'result-relevance': result.relevance_score || '-',
                    'result-events': result.events_count || '-',
                    'analysis-time': formatDuration(processingTime)
                };
                
                Object.entries(elements).forEach(([id, value]) => {
                    const element = document.getElementById(id);
                    if (element) element.textContent = value;
                });
                
                // Update segment time range display
                const segmentTimeRange = document.getElementById('segment-time-range');
                if (segmentTimeRange) {
                    segmentTimeRange.textContent = `${this.formatTime(startTime)} to ${this.formatTime(endTime)}`;
                }
                
                // Store segment timing for playback
                const playSegmentBtn = document.getElementById('play-segment-btn');
                if (playSegmentBtn) {
                    playSegmentBtn.dataset.startTime = startTime;
                    playSegmentBtn.dataset.endTime = endTime;
                    playSegmentBtn.onclick = () => this.playVideoSegment(startTime, endTime);
                }
                
                // Show completion notification with timing
                showNotification(
                    `✅ Analysis completed in ${formatDuration(processingTime)}`, 
                    'success', 
                    4000
                );
            } else {
                throw new Error(result.error || 'No chunk found or analysis failed');
            }
        } catch (error) {
            console.error('Error finding relevant chunk:', error);
            const processingTime = analysisTimer.end();
            if (loadingDiv) loadingDiv.style.display = 'none';
            if (placeholderDiv) {
                placeholderDiv.style.display = 'block';
                placeholderDiv.innerHTML = `
                    <i class="fas fa-exclamation-triangle" style="font-size: 3rem; opacity: 0.3; color: var(--danger-color);"></i>
                    <p style="margin-top: 1rem;">Error analyzing chunks</p>
                    <p style="font-size: 0.875rem;">Failed after ${formatDuration(processingTime)}</p>
                `;
            }
            
            showNotification(
                `❌ Analysis failed after ${formatDuration(processingTime)}`, 
                'error', 
                6000
            );
        }
    }

    playVideoSegment(startTime, endTime) {
        const videoPlayer = document.getElementById('analysis-preview-player');
        const currentVideoFile = document.getElementById('current-video-file');
        
        if (!videoPlayer || !currentVideoFile || !currentVideoFile.value) {
            showNotification('❌ No video loaded for playback', 'error', 3000);
            return;
        }
        
        // Ensure video is loaded and ready
        if (videoPlayer.src !== `/serve-video/${currentVideoFile.value}`) {
            videoPlayer.src = `/serve-video/${currentVideoFile.value}`;
            videoPlayer.load();
        }
        
        // Set up time update event listener to stop at end time
        const handleTimeUpdate = () => {
            if (videoPlayer.currentTime >= endTime) {
                videoPlayer.pause();
                videoPlayer.removeEventListener('timeupdate', handleTimeUpdate);
                showNotification('⏹️ Segment playback completed', 'info', 2000);
            }
        };
        
        // Remove any existing listener
        videoPlayer.removeEventListener('timeupdate', handleTimeUpdate);
        
        // Add new listener and start playback
        videoPlayer.addEventListener('timeupdate', handleTimeUpdate);
        
        // Seek to start time and play
        videoPlayer.currentTime = startTime;
        videoPlayer.play()
            .then(() => {
                showNotification(`▶️ Playing segment: ${this.formatTime(startTime)} to ${this.formatTime(endTime)}`, 'success', 2000);
            })
            .catch(error => {
                console.error('Playback failed:', error);
                showNotification('❌ Failed to play video segment', 'error', 3000);
                videoPlayer.removeEventListener('timeupdate', handleTimeUpdate);
            });
    }

    async performTemporalAnalysis(videoFile, query, startTime, endTime) {
        const loadingDiv = document.getElementById('temporal-analysis-loading');
        const resultsDiv = document.getElementById('temporal-analysis-results');
        const placeholderDiv = document.getElementById('temporal-analysis-placeholder');
        
        // Show loading state
        const analysisTimer = this.startTiming();
        if (loadingDiv) loadingDiv.style.display = 'block';
        if (resultsDiv) resultsDiv.style.display = 'none';
        
        // Hide chunk analysis results and show temporal results
        document.getElementById('chunk-analysis-results').classList.add('hidden');
        
        // Show start notification
        showNotification('🔍 Starting temporal analysis...', 'info', 3000);
        
        try {
            const response = await fetch('/api/temporal-analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_file: videoFile,
                    query: query,
                    start_time: startTime,
                    end_time: endTime
                })
            });
            
            const result = await response.json();
            const processingTime = analysisTimer.end();
            
            // Debug: Log the response structure
            console.log('Temporal analysis response:', result);
            console.log('Response keys:', Object.keys(result));
            console.log('Success:', result.success);
            console.log('Has temporal_analysis:', 'temporal_analysis' in result);
            console.log('Has key_events:', 'key_events' in result);
            
            if (result.success) {
                // Debug: Check if elements exist
                console.log('Loading div:', loadingDiv);
                console.log('Results div:', resultsDiv);
                
                // Hide loading and show results
                if (loadingDiv) loadingDiv.style.display = 'none';
                if (resultsDiv) {
                    resultsDiv.style.display = 'block';
                    resultsDiv.classList.remove('hidden');  // Remove hidden class that overrides display
                }
                
                const duration = endTime - startTime;
                
                // Update temporal analysis displays
                const elements = {
                    'temporal-time-range': `${this.formatTime(startTime)} to ${this.formatTime(endTime)}`,
                    'temporal-duration': `${duration.toFixed(1)}s`,
                    'temporal-frames': result.frames_analyzed || result.num_frames || '-',
                    'temporal-events': result.key_events ? result.key_events.length : (result.events_detected || '-'),
                    'temporal-analysis-content': result.temporal_analysis || result.analysis || 'No analysis available'
                };
                
                Object.entries(elements).forEach(([id, value]) => {
                    const element = document.getElementById(id);
                    console.log(`Updating element ${id}:`, element, 'with value:', value);
                    if (element) {
                        element.textContent = value;
                    } else {
                        console.warn(`Element with id '${id}' not found`);
                    }
                });
                
                // Display key events if available
                const keyEventsDiv = document.getElementById('temporal-key-events');
                if (keyEventsDiv && result.key_events && result.key_events.length > 0) {
                    keyEventsDiv.innerHTML = `
                        <h4>Key Events Timeline:</h4>
                        <ul class="key-events-list">
                            ${result.key_events.map(event => `
                                <li>
                                    <strong>${event.timestamp ? this.formatTime(event.timestamp) : 'Unknown time'}:</strong>
                                    ${event.description || 'No description'}
                                </li>
                            `).join('')}
                        </ul>
                    `;
                    keyEventsDiv.style.display = 'block';
                } else if (keyEventsDiv) {
                    keyEventsDiv.style.display = 'none';
                }
                
                // Setup play temporal segment button
                const playTemporalBtn = document.getElementById('play-temporal-btn');
                if (playTemporalBtn) {
                    playTemporalBtn.onclick = () => this.playVideoSegment(startTime, endTime);
                    document.getElementById('temporal-segment-info').textContent = 
                        `${duration.toFixed(1)}s segment ready`;
                }
                
                // Show completion notification
                showNotification(
                    `✅ Temporal analysis completed in ${formatDuration(processingTime)}`, 
                    'success', 
                    4000
                );
            } else {
                throw new Error(result.error || 'Temporal analysis failed');
            }
        } catch (error) {
            console.error('Error performing temporal analysis:', error);
            const processingTime = analysisTimer.end();
            
            showNotification(
                `❌ Temporal analysis failed after ${formatDuration(processingTime)}`, 
                'error', 
                6000
            );
        }
    }

    async showTimelineForAnalysis(videoFile) {
        try {
            showNotification('Loading timeline for analysis...', 'info');

            // Fetch stored caption data for the selected video
            const response = await fetch(`/api/video-caption/${encodeURIComponent(videoFile)}`);
            if (!response.ok) {
                throw new Error('Failed to load caption data');
            }

            const result = await response.json();
            if (!result.success || !result.data) {
                throw new Error('No caption data available');
            }

            // Set up global variables for timeline use
            currentVideoData = {
                filename: videoFile,
                duration: result.data.video_duration || 0,
                fps_used: result.data.fps_used || 3.0,
                processing_mode: result.data.analysis_method || 'unknown'
            };

            // Convert stored caption data to format expected by timeline
            currentCaptionData = result.data.chunk_details?.map(chunk => ({
                chunk_id: chunk.chunk_id || 1,
                start_time: chunk.start_time || 0,
                end_time: chunk.end_time || (chunk.start_time || 0) + 30,
                caption: chunk.caption || ''
            })) || [];

            if (currentCaptionData.length === 0) {
                showNotification('No caption data available for timeline generation', 'warning');
                return;
            }

            // Switch to caption page and show timeline
            this.switchToCaptionPage();

            // Wait a moment for the page to switch, then show timeline
            setTimeout(() => {
                showTimeline();
            }, 500);

            showNotification('Timeline loaded successfully', 'success');

        } catch (error) {
            console.error('Error showing timeline for analysis:', error);
            showNotification(`Failed to load timeline: ${error.message}`, 'error');
        }
    }

    switchToCaptionPage() {
        // Switch to the caption generation page
        const captionPage = document.querySelector('[data-page="caption"]');
        const analysisPage = document.querySelector('[data-page="analysis"]');

        if (captionPage && analysisPage) {
            // Update navigation
            document.querySelectorAll('.sidebar-compact .nav-link').forEach(link => {
                link.classList.remove('active');
            });

            const captionLink = document.querySelector('[data-page="caption"]');
            if (captionLink) {
                captionLink.classList.add('active');
            }

            // Hide analysis page and show caption page
            analysisPage.classList.add('hidden');
            captionPage.classList.remove('hidden');

            // Update page title
            document.title = 'InternVideo2.5 Surveillance System - Caption Generation';
        }
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatDuration(milliseconds) {
        if (milliseconds < 1000) return `${milliseconds}ms`;
        const seconds = milliseconds / 1000;
        if (seconds < 60) return `${seconds.toFixed(1)}s`;
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    startTiming() {
        return {
            startTime: performance.now(),
            getElapsed: () => performance.now() - this.startTime,
            end: () => performance.now() - this.startTime
        };
    }

    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        
        const icons = {
            success: 'check-circle',
            error: 'exclamation-triangle', 
            warning: 'exclamation-circle',
            info: 'info-circle',
            timing: 'clock'
        };
        
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <i class="fas fa-${icons[type] || icons.info}"></i>
                <div>
                    <div style="font-weight: 500;">${message}</div>
                    <div style="font-size: 0.75rem; opacity: 0.7; margin-top: 0.25rem;">
                        ${new Date().toLocaleTimeString()}
                    </div>
                </div>
                <button onclick="this.closest('.notification').remove()" style="
                    margin-left: auto;
                    background: none;
                    border: none;
                    color: inherit;
                    cursor: pointer;
                    opacity: 0.7;
                    font-size: 1.2rem;
                    line-height: 1;
                ">×</button>
            </div>
        `;
        
        // Enhanced positioning with backdrop
        notification.style.cssText = `
            position: fixed;
            top: 2rem;
            right: 2rem;
            z-index: 10000;
            min-width: 320px;
            max-width: 500px;
            animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
            background: rgba(255, 255, 255, 0.95);
        `;
        
        document.body.appendChild(notification);
        
        // Auto-dismiss with enhanced animations
        const dismissTimer = setTimeout(() => {
            this.dismissNotification(notification);
        }, duration);
        
        // Click to dismiss
        notification.addEventListener('click', (e) => {
            if (!e.target.closest('button')) {
                clearTimeout(dismissTimer);
                this.dismissNotification(notification);
            }
        });
        
        return notification;
    }

    dismissNotification(notification) {
        if (notification && notification.parentNode) {
            notification.style.animation = 'slideOut 0.3s cubic-bezier(0.4, 0, 0.2, 1) forwards';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 300);
        }
    }

    showSpinner(element) {
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        element.appendChild(spinner);
        return spinner;
    }

    hideSpinner(spinner) {
        if (spinner && spinner.parentNode) {
            spinner.parentNode.removeChild(spinner);
        }
    }
}


// Caption Generator Handler
class CaptionGenerator {
    constructor() {
        this.isGenerating = false;
    }

    init() {
        this.setupEventListeners();
        this.loadAvailableVideos();

        // Set up periodic check for timeline button
        setInterval(() => {
            this.checkAndShowTimelineButton();
        }, 2000); // Check every 2 seconds
    }

    setupEventListeners() {
        const generateBtn = document.querySelector('#generate-caption-btn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.generateCaption());
        }

        // Add event listeners for time estimation
        const videoSelect = document.querySelector('#caption-video-select');
        const fpsInput = document.querySelector('#caption-fps');
        const chunkInput = document.querySelector('#chunk-size');
        const modeSelect = document.querySelector('#processing-mode');

        if (videoSelect) {
            videoSelect.addEventListener('change', () => this.updateTimeEstimation());
        }
        if (fpsInput) {
            fpsInput.addEventListener('input', () => this.updateTimeEstimation());
        }
        if (chunkInput) {
            chunkInput.addEventListener('input', () => this.updateTimeEstimation());
        }
        if (modeSelect) {
            modeSelect.addEventListener('change', () => this.updateTimeEstimation());
        }
    }

    checkAndShowTimelineButton() {
        // Check if we have caption data and show timeline button if available
        if (currentCaptionData && currentCaptionData.length > 0) {
            const timelineBtn = document.getElementById('caption-page-timeline-btn');
            if (timelineBtn) {
                timelineBtn.classList.remove('hidden');
            }
        }
    }

    async loadAvailableVideos() {
        try {
            const response = await fetch('/available-videos');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            const videoSelect = document.querySelector('#caption-video-select');
            if (!videoSelect) {
                console.warn('Caption video select element not found');
                return;
            }

            if (data.error) {
                videoSelect.innerHTML = '<option value="">Error loading videos</option>';
                console.error('Server error:', data.error);
                return;
            }
            
            if (data.videos && Array.isArray(data.videos)) {
                videoSelect.innerHTML = '<option value="">Select a video...</option>';
                
                data.videos.forEach(video => {
                    if (video.filename) {
                        const option = document.createElement('option');
                        option.value = video.filename;
                        option.textContent = `${video.filename} (${video.size_mb || '?'}MB, ${video.duration || 'Unknown'})`;
                        videoSelect.appendChild(option);
                    }
                });
                
                console.log(`Caption generator: Loaded ${data.videos.length} videos successfully`);

                // Check if timeline button should be shown
                this.checkAndShowTimelineButton();
            } else {
                videoSelect.innerHTML = '<option value="">No videos found</option>';
                console.warn('No videos data received for caption generator');
            }
        } catch (error) {
            console.error('Caption generator: Failed to load videos:', error);
            const videoSelect = document.querySelector('#caption-video-select');
            if (videoSelect) {
                videoSelect.innerHTML = '<option value="">Failed to load videos</option>';
            }

            if (window.showNotification) {
                window.showNotification('Failed to load video list for caption generator', 'error');
            }
        }
    }

    async generateCaption() {
        if (this.isGenerating) return;

        const videoSelect = document.querySelector('#caption-video-select');
        const fpsInput = document.querySelector('#caption-fps');
        const chunkSize = document.querySelector('#chunk-size');
        const promptInput = document.querySelector('#caption-prompt');
        const modeSelect = document.querySelector('#processing-mode');
        const outputFormat = document.querySelector('#output-format-select');

        if (!videoSelect.value) {
            alert('Please select a video first');
            return;
        }

        this.isGenerating = true;
        const generateBtn = document.querySelector('#generate-caption-btn');
        const originalText = generateBtn.textContent;
        
        generateBtn.textContent = 'Generating...';
        generateBtn.disabled = true;

        // Initialize real-time results display
        this.initializeRealTimeResults();

        // Store current video data for timeline use
        currentVideoData = {
            filename: videoSelect.value,
            fps_sampling: parseFloat(fpsInput.value) || 1.0,
            chunk_size: parseInt(chunkSize.value) || 60,
            processing_mode: modeSelect.value || 'sequential'
        };

        // Reset caption data for new generation
        currentCaptionData = [];

        // Hide timeline view and show caption results
        const timelineView = document.getElementById('timeline-view');
        const captionResults = document.getElementById('caption-results');
        if (timelineView) timelineView.classList.add('hidden');
        if (captionResults) captionResults.classList.remove('hidden');

        const requestData = {
            video_file: videoSelect.value,
            fps_sampling: parseFloat(fpsInput.value) || 1.0,
            chunk_size: parseInt(chunkSize.value) || 60,
            prompt: promptInput.value.trim() || 'Generate a detailed caption describing what happens in this video.',
            processing_mode: modeSelect.value || 'sequential',
            output_format: outputFormat ? outputFormat.value : 'paragraph'
        };

        try {
            console.log('🚀 Starting caption generation with GPT:', {
                videoFile: requestData.video_file,
                prompt: requestData.prompt,
                fps: requestData.fps_sampling,
                chunkSize: requestData.chunk_size,
                mode: requestData.processing_mode
            });

            // Use Server-Sent Events for real-time updates
            const response = await fetch('/generate-caption-stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    const trimmedLine = line.trim();
                    if (trimmedLine.startsWith('data: ')) {
                        try {
                            const jsonData = trimmedLine.slice(6); // Remove 'data: ' prefix
                            if (jsonData) { // Only parse if there's actual data
                                const data = JSON.parse(jsonData);
                                this.handleRealTimeUpdate(data);
                            }
                        } catch (e) {
                            console.warn('Failed to parse SSE data:', trimmedLine, 'Error:', e.message);
                        }
                    }
                    // Skip empty lines and event lines
                    if (trimmedLine === '' || trimmedLine.startsWith('event:')) {
                        continue;
                    }
                }
            }

        } catch (error) {
            console.error('Caption generation error:', error);
            const resultsEl = document.querySelector('#caption-results');
            resultsEl.innerHTML = `<div class="status-indicator status-error">Error: ${error.message}</div>`;
        } finally {
            this.isGenerating = false;
            generateBtn.textContent = originalText;
            generateBtn.disabled = false;
        }
    }

    initializeRealTimeResults() {
        const resultsEl = document.querySelector('#caption-results');
        resultsEl.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🚀 Real-Time Caption Generation</div>
                    <div class="card-subtitle">Processing chunks as they complete...</div>
                </div>
                <div class="card-body">
                    <div id="progress-section">
                        <div class="progress-bar">
                            <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
                        </div>
                        <div id="progress-text">Initializing...</div>
                    </div>
                    
                    <div id="live-chunks" class="mt-3">
                        <!-- Chunks will be added here as they complete -->
                    </div>
                    
                </div>
            </div>
        `;
        resultsEl.classList.remove('hidden');
    }

    handleRealTimeUpdate(data) {
        const progressText = document.querySelector('#progress-text');
        const progressFill = document.querySelector('#progress-fill');
        const captionResults = document.querySelector('#caption-results');
        
        switch (data.type) {
            case 'status':
                progressText.textContent = data.message;
                break;
                
            case 'processing_info':
                this.captionStartTime = performance.now();
                progressText.textContent = `Processing ${data.total_chunks} chunks with ${data.total_frames} total frames`;
                showNotification(`🚀 Started processing ${data.total_chunks} chunks`, 'info', 3000);
                break;
                
            case 'chunk_start':
                this.currentChunkStart = performance.now();
                progressText.textContent = `Processing chunk ${data.chunk}/${data.total_chunks} (${data.start_time.toFixed(1)}s - ${data.end_time.toFixed(1)}s)`;
                break;
                
            case 'chunk_complete':
                // Log GPT response received
                console.log('🤖 GPT Response Received:', {
                    chunk: data.chunk,
                    timestamp: `${data.start_time.toFixed(1)}s - ${data.end_time.toFixed(1)}s`,
                    caption: data.caption,
                    processingTime: data.processing_time,
                    characters: data.characters
                });

                // Update progress
                const progress = data.progress || 0;
                progressFill.style.width = `${progress}%`;

                // Calculate chunk timing
                const chunkTime = this.currentChunkStart ? performance.now() - this.currentChunkStart : 0;
                const totalTime = this.captionStartTime ? performance.now() - this.captionStartTime : 0;

                progressText.textContent = `Completed chunk ${data.chunk}/${data.total_chunks} (${progress.toFixed(1)}%) - ${formatDuration(totalTime)} total`;

                // Make caption results visible
                captionResults.classList.remove('hidden');

                // Add chunk result to display with enhanced timing
                const chunkDiv = document.createElement('div');
                chunkDiv.className = 'chunk-result';

                // Ensure caption is displayed even if empty
                let captionText = data.caption || 'No caption text available';
                if (typeof captionText !== 'string') {
                    captionText = JSON.stringify(captionText);
                }

                chunkDiv.innerHTML = `
                    <div class="chunk-header">
                        <strong>📹 Chunk ${data.chunk}/${data.total_chunks}</strong>
                        <span class="chunk-timing">
                            ⏱️ ${formatDuration(chunkTime)} • ${data.characters || 0} chars
                            ${data.processing_time ? `• Server: ${formatDuration(data.processing_time * 1000)}` : ''}
                        </span>
                    </div>
                    <div class="chunk-content">${captionText}</div>
                `;
                captionResults.appendChild(chunkDiv);
                captionResults.scrollTop = captionResults.scrollHeight; // Auto-scroll

                // Store caption data for timeline use
                if (!currentCaptionData) {
                    currentCaptionData = [];
                }
                currentCaptionData.push({
                    chunk_id: data.chunk_id || data.chunk,
                    start_time: data.start_time,
                    end_time: data.end_time,
                    caption: data.caption,
                    processing_time: data.processing_time,
                    characters: data.characters || 0
                });

                // Log for debugging
                console.log('Chunk completed:', data);
                break;
                
            case 'cleanup':
                // Show cleanup notifications briefly
                const cleanupMsg = document.createElement('div');
                cleanupMsg.className = 'cleanup-msg';
                cleanupMsg.textContent = `🧹 ${data.message}`;
                cleanupMsg.style.opacity = '0.7';
                cleanupMsg.style.fontSize = '0.8rem';
                captionResults.appendChild(cleanupMsg);
                setTimeout(() => cleanupMsg.remove(), 3000);
                break;
                
            case 'complete':
                // Calculate total processing time
                const finalTotalTime = this.captionStartTime ? performance.now() - this.captionStartTime : 0;

                progressText.textContent = `✅ Completed! ${data.chunks_processed} chunks in ${formatDuration(finalTotalTime)}`;
                progressFill.style.width = '100%';

                // Store additional video metadata from the completion data
                if (data.video_duration && currentVideoData) {
                    currentVideoData.duration = data.video_duration;
                }
                if (data.frames_processed && currentVideoData) {
                    currentVideoData.frames_processed = data.frames_processed;
                }
                if (data.analysis_method && currentVideoData) {
                    currentVideoData.processing_mode = data.analysis_method;
                }

                // Show completion notification
                showNotification(
                    `✅ Caption generation completed in ${formatDuration(finalTotalTime)}`,
                    'success',
                    6000
                );

                // Show timeline button in caption page
                const timelineBtn = document.getElementById('caption-page-timeline-btn');
                if (timelineBtn && currentCaptionData && currentCaptionData.length > 0) {
                    timelineBtn.classList.remove('hidden');
                    showNotification('Timeline button is now available', 'info');
                }
                break;
                
            case 'error':
            case 'chunk_error':
                console.error('Processing error:', data);
                const errorTime = this.captionStartTime ? performance.now() - this.captionStartTime : 0;
                progressText.innerHTML = `<span style="color: red;">Error after ${formatDuration(errorTime)}: ${data.message || data.error}</span>`;
                
                showNotification(
                    `❌ Processing error after ${formatDuration(errorTime)}`, 
                    'error', 
                    8000
                );
                break;
        }
    }

    showCaptionResults(data) {
        const resultsEl = document.querySelector('#caption-results');
        const captionText = data.caption.replace(/'/g, "&apos;").replace(/"/g, "&quot;");
        
        resultsEl.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <div class="card-title">✅ Caption Generated Successfully</div>
                    <div class="card-subtitle">
                        ${data.video_file} | ${data.processing_mode} mode | ${data.output_format} format
                    </div>
                </div>
                <div class="card-body">
                    <div class="form-group">
                        <label class="form-label">Generated Caption:</label>
                        <textarea class="form-textarea" rows="12" readonly>${data.caption}</textarea>
                    </div>
                    
                    <div class="grid grid-3">
                        <div>
                            <strong>📊 Chunks Processed:</strong><br>
                            <span style="color: var(--text-secondary);">${data.chunks_processed}</span>
                        </div>
                        <div>
                            <strong>🎯 Frames Analyzed:</strong><br>
                            <span style="color: var(--text-secondary);">${data.frames_processed}</span>
                        </div>
                        <div>
                            <strong>⚡ FPS Sampling:</strong><br>
                            <span style="color: var(--text-secondary);">${data.fps_used}</span>
                        </div>
                        <div>
                            <strong>⏱️ Processing Time:</strong><br>
                            <span style="color: var(--text-secondary);">${data.processing_time}</span>
                        </div>
                        <div>
                            <strong>📏 Caption Length:</strong><br>
                            <span style="color: var(--text-secondary);">${data.caption.length} characters</span>
                        </div>
                        <div>
                            <strong>📅 Generated:</strong><br>
                            <span style="color: var(--text-secondary);">${new Date(data.timestamp).toLocaleTimeString()}</span>
                        </div>
                    </div>
                    
                    <div class="mt-3">
                        <button class="btn btn-primary" onclick="window.copyToClipboard('${captionText}')">
                            <i class="fas fa-copy"></i> Copy Caption
                        </button>
                        <button class="btn btn-outline ml-2" onclick="window.downloadCaptionFile('${data.video_file}', '${captionText}')"
                            <i class="fas fa-download"></i> Download TXT
                        </button>
                        <button class="btn btn-outline ml-2" onclick="window.location.reload()">
                            <i class="fas fa-redo"></i> Generate Another
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    async updateTimeEstimation() {
        const videoSelect = document.querySelector('#caption-video-select');
        const fpsInput = document.querySelector('#caption-fps');
        const chunkInput = document.querySelector('#chunk-size');
        const modeSelect = document.querySelector('#processing-mode');
        const estimationEl = document.querySelector('#time-estimation');

        if (!estimationEl) return;

        if (!videoSelect.value) {
            estimationEl.innerHTML = 'Select a video to see processing time estimate';
            return;
        }

        const estimationTimer = {
            startTime: performance.now(),
            end: () => performance.now() - this.startTime
        };
        estimationEl.innerHTML = '<div class="spinner"></div> Calculating estimate...';

        try {
            const response = await fetch('/caption-estimate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_file: videoSelect.value,
                    fps_sampling: parseFloat(fpsInput.value) || 1.0,
                    chunk_size: parseInt(chunkInput.value) || 60,
                    processing_mode: modeSelect.value || 'sequential'
                })
            });

            const data = await response.json();
            const calcTime = estimationTimer.end();
            
            if (data.success) {
                const chunks = data.processing_info.chunks_count;
                const totalFrames = data.processing_info.total_frames_to_process;
                const duration = data.video_info.duration;
                const processingMode = data.processing_info.processing_mode;
                
                let modeDescription = '';
                switch (processingMode) {
                    case 'overlapping':
                        modeDescription = ' (25% overlap)';
                        break;
                    case 'keyframe':
                        modeDescription = ' (keyframe detection)';
                        break;
                    case 'adaptive':
                        modeDescription = ' (adaptive sampling)';
                        break;
                }

                estimationEl.innerHTML = `
                    <div class="estimation-header">
                        <strong>⏱️ Processing Estimate</strong>
                        <span class="calc-time">Calculated in ${formatDuration(calcTime)}</span>
                    </div>
                    <div class="estimation-grid">
                        <div class="estimation-item">
                            <span class="estimation-label">📹 Video Duration</span>
                            <span class="estimation-value">${duration}s (${Math.round(data.video_info.fps)} FPS)</span>
                        </div>
                        <div class="estimation-item">
                            <span class="estimation-label">📊 Processing Mode</span>
                            <span class="estimation-value">${chunks} chunks${modeDescription}</span>
                        </div>
                        <div class="estimation-item">
                            <span class="estimation-label">🎯 Total Frames</span>
                            <span class="estimation-value">${totalFrames} (~${Math.round(totalFrames/chunks)}/chunk)</span>
                        </div>
                        <div class="estimation-item">
                            <span class="estimation-label">⏱️ Estimated Time</span>
                            <span class="estimation-value">~${data.time_estimation.total_time_minutes} minutes</span>
                        </div>
                    </div>
                    <div class="estimation-details">
                        <div>Processing: ${data.time_estimation.processing_time_seconds}s + Model loading: 16s</div>
                        <div>Average: ${formatDuration((data.time_estimation.processing_time_seconds * 1000) / chunks)} per chunk</div>
                    </div>
                `;

                // Show chunk preview if available
                if (data.chunk_details && data.chunk_details.length > 0) {
                    const previewHtml = data.chunk_details.slice(0, 3).map(chunk => 
                        `Chunk ${chunk.chunk_number}: ${chunk.start_time.toFixed(1)}s-${chunk.end_time.toFixed(1)}s (${chunk.frames} frames)`
                    ).join('<br>');
                    
                    estimationEl.innerHTML += `
                        <div class="estimation-preview">
                            <strong>Preview:</strong><br>
                            ${previewHtml}
                            ${data.chunk_details.length > 3 ? '<br>...' : ''}
                        </div>
                    `;
                }
            } else {
                estimationEl.innerHTML = `<div class="estimation-error">Error: ${data.error}</div>`;
            }
        } catch (error) {
            estimationEl.innerHTML = `<div class="estimation-error">Failed to calculate estimate</div>`;
            console.error('Estimation error:', error);
        }
    }

    downloadCaption(videoFile, caption) {
        const blob = new Blob([caption], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${videoFile}_caption.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}


// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.SurveillanceApp = new SurveillanceApp();
    window.CaptionGenerator = new CaptionGenerator();
    window.CaptionGenerator.init();
});