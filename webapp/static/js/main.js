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
        // Hide all page content
        document.querySelectorAll('.page-content').forEach(content => {
            content.classList.add('hidden');
        });

        // Show selected page
        const targetPage = document.querySelector(`#${page}-page`);
        if (targetPage) {
            targetPage.classList.remove('hidden');
        }

        // Load page-specific content if needed
        switch (page) {
            case 'caption':
                this.loadCaptionPage();
                break;
            case 'analysis':
                this.loadAnalysisPage();
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
    }

    loadAnalysisPage() {
        // Initialize analysis functionality
        console.log('Analysis page loaded');
        this.setupAnalysisPage();
    }

    async setupAnalysisPage() {
        // Load stored captions for video selection
        await this.loadStoredCaptionsForAnalysis();
        
        // Setup event listeners
        const videoSelector = document.getElementById('video-selector');
        const analyzeBtn = document.getElementById('analyze-chunks-btn');
        
        if (videoSelector) {
            videoSelector.addEventListener('change', async (e) => {
                const videoFile = e.target.value;
                if (videoFile) {
                    await this.displayVideoInfoForAnalysis(videoFile);
                    await this.loadChunkDataForAnalysis(videoFile);
                    if (analyzeBtn) analyzeBtn.disabled = false;
                } else {
                    this.hideVideoInfoDisplay();
                    if (analyzeBtn) analyzeBtn.disabled = true;
                }
            });
        }

        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', async () => {
                const videoFile = videoSelector?.value;
                const query = document.getElementById('chunk-query')?.value?.trim();
                if (videoFile && query) {
                    await this.findRelevantChunk(videoFile, query);
                }
            });
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
            const response = await fetch(`/api/video-caption/${videoFile}`);
            const result = await response.json();
            const data = result.data || result; // Handle nested data structure
            
            if (data && result.success !== false) {
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
            const response = await fetch(`/api/video-caption/${videoFile}`);
            const result = await response.json();
            const data = result.data || result; // Handle nested data structure
            
            const chunkDataTextarea = document.getElementById('chunk-data');
            if (chunkDataTextarea && data.chunks) {
                // Format chunks for display
                const formattedChunks = data.chunks.map(chunk => ({
                    chunk_id: chunk.chunk_id || 1,
                    start_time: chunk.start_time || 0,
                    end_time: chunk.end_time || 0,
                    caption: chunk.caption || ''
                }));
                chunkDataTextarea.value = JSON.stringify(formattedChunks, null, 2);
            }
        } catch (error) {
            console.error('Error loading chunk data:', error);
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
            const response = await fetch('/api/find-relevant-chunk', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_file: videoFile,
                    query: query
                })
            });
            
            const result = await response.json();
            const processingTime = analysisTimer.end();
            
            if (result.chunk_info) {
                // Hide loading and show results
                if (loadingDiv) loadingDiv.style.display = 'none';
                if (resultsDiv) resultsDiv.style.display = 'block';
                
                // Calculate and display duration properly
                const startTime = result.chunk_info.start_time || 0;
                const endTime = result.chunk_info.end_time || 0;
                const duration = endTime - startTime;
                
                // Update result displays with proper formatting
                const elements = {
                    'result-chunk-number': result.chunk_info.chunk_id || '-',
                    'result-start-time': this.formatTime(startTime),
                    'result-end-time': this.formatTime(endTime),
                    'result-duration': duration.toFixed(1),
                    'result-caption': result.chunk_info.caption || '-',
                    'analysis-time': formatDuration(processingTime)
                };
                
                Object.entries(elements).forEach(([id, value]) => {
                    const element = document.getElementById(id);
                    if (element) element.textContent = value;
                });
                
                // Show completion notification with timing
                showNotification(
                    `✅ Analysis completed in ${formatDuration(processingTime)}`, 
                    'success', 
                    4000
                );
            } else {
                throw new Error('No chunk found');
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

        const requestData = {
            video_file: videoSelect.value,
            fps_sampling: parseFloat(fpsInput.value) || 1.0,
            chunk_size: parseInt(chunkSize.value) || 60,
            prompt: promptInput.value.trim() || 'Generate a detailed caption describing what happens in this video.',
            processing_mode: modeSelect.value || 'sequential',
            output_format: outputFormat ? outputFormat.value : 'paragraph'
        };

        try {
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
                    
                    <div id="final-caption" class="hidden mt-4">
                        <!-- Final combined caption will appear here -->
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
                
                // Show final combined caption with timing
                const finalDiv = document.querySelector('#final-caption');
                finalDiv.innerHTML = `
                    <div class="final-caption-header">
                        <h3>✅ Final Caption</h3>
                        <div class="caption-stats">
                            ${data.caption.length} characters • ${data.chunks_processed} chunks • ${data.total_frames} frames
                            <br>
                            <span class="timing-badge">⏱️ Total: ${formatDuration(finalTotalTime)}</span>
                        </div>
                    </div>
                    <div class="final-caption-content">${data.caption}</div>
                `;
                finalDiv.classList.remove('hidden');
                progressText.textContent = `✅ Completed! ${data.chunks_processed} chunks in ${formatDuration(finalTotalTime)}`;
                progressFill.style.width = '100%';
                
                // Show completion notification
                showNotification(
                    `✅ Caption generation completed in ${formatDuration(finalTotalTime)}`, 
                    'success', 
                    6000
                );
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