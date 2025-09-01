// InternVideo2.5 Surveillance - Main JavaScript

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
        // Load the default page (chat interface)
        const defaultNavItem = document.querySelector('.nav-link[data-page="chat"]');
        if (defaultNavItem) {
            this.navigateToPage('chat');
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
            case 'chat':
                this.loadChatPage();
                break;
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

    loadChatPage() {
        // Initialize chat functionality
        if (window.ChatInterface) {
            window.ChatInterface.init();
        }
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
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
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

// Chat Interface Handler
class ChatInterface {
    constructor() {
        this.isProcessing = false;
        this.currentVideo = null;
    }

    init() {
        this.setupEventListeners();
        this.loadAvailableVideos();
        this.checkSystemStatus();
    }

    setupEventListeners() {
        const processBtn = document.querySelector('#process-video-btn');
        const chatForm = document.querySelector('#chat-form');
        const clearBtn = document.querySelector('#clear-chat-btn');

        if (processBtn) {
            processBtn.addEventListener('click', () => this.processVideo());
        }

        if (chatForm) {
            chatForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.sendMessage();
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearChat());
        }
    }

    async loadAvailableVideos() {
        try {
            const response = await fetch('/available-videos');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            const videoSelect = document.querySelector('#video-select');
            if (!videoSelect) {
                console.warn('Video select element not found');
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
                
                console.log(`Loaded ${data.videos.length} videos successfully`);
            } else {
                videoSelect.innerHTML = '<option value="">No videos found</option>';
                console.warn('No videos data received');
            }
        } catch (error) {
            console.error('Failed to load videos:', error);
            const videoSelect = document.querySelector('#video-select');
            if (videoSelect) {
                videoSelect.innerHTML = '<option value="">Failed to load videos</option>';
            }
            
            // Show user notification
            if (window.showNotification) {
                window.showNotification('Failed to load video list', 'error');
            }
        }
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            const statusEl = document.querySelector('#system-status');
            if (statusEl) {
                statusEl.textContent = data.status || 'unknown';
                statusEl.className = `status-indicator ${data.status === 'online' ? 'status-success' : 'status-error'}`;
            }
        } catch (error) {
            console.error('Failed to check status:', error);
        }
    }

    async processVideo() {
        if (this.isProcessing) return;

        const videoSelect = document.querySelector('#video-select');
        const numFrames = document.querySelector('#num-frames');
        const fpsInput = document.querySelector('#fps-sampling');
        const timeBound = document.querySelector('#time-bound');

        if (!videoSelect.value) {
            window.showNotification('Please select a video first', 'warning');
            return;
        }

        this.isProcessing = true;
        const processBtn = document.querySelector('#process-video-btn');
        const originalText = processBtn.textContent;
        const statusEl = document.querySelector('#processing-status');
        
        processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        processBtn.disabled = true;
        
        // Show processing status
        statusEl.classList.remove('hidden');
        statusEl.innerHTML = '<div class="status-indicator status-info"><i class="fas fa-spinner fa-spin"></i> Processing video frames...</div>';

        try {
            const requestData = {
                video_file: videoSelect.value,
                num_frames: parseInt(numFrames.value) || 64,
                fps_sampling: parseFloat(fpsInput.value) || null
            };

            // Add time bound if specified
            if (timeBound.value) {
                requestData.time_bound = parseFloat(timeBound.value);
            }

            const response = await fetch('/process-video', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();
            
            if (data.success) {
                this.currentVideo = videoSelect.value;
                this.showProcessingResults(data);
                this.enableChat();
                this.updateVideoPlayer(videoSelect.value);
                
                
                // Trigger custom event for other components
                window.dispatchEvent(new CustomEvent('videoProcessed', { detail: data }));
            } else {
                window.showNotification(data.error || 'Processing failed', 'error');
                statusEl.innerHTML = '<div class="status-indicator status-error">Processing failed</div>';
            }
        } catch (error) {
            window.showNotification('Error processing video: ' + error.message, 'error');
            statusEl.innerHTML = '<div class="status-indicator status-error">Processing error</div>';
        } finally {
            this.isProcessing = false;
            processBtn.innerHTML = originalText;
            processBtn.disabled = false;
        }
    }

    showProcessingResults(data) {
        const resultsEl = document.querySelector('#processing-results');
        const statusEl = document.querySelector('#processing-status');
        
        // Show compact notification at top-right instead of inline results
        if (window.showNotification) {
            window.showNotification(
                `✅ <strong>Video Ready!</strong><br>📊 ${data.frames_processed} frames • ${data.processing_time}`,
                'success',
                6000
            );
        }
        
        // Hide both results and status elements to keep UI clean
        if (resultsEl) {
            resultsEl.classList.add('hidden');
        }
        if (statusEl) {
            statusEl.classList.add('hidden');
        }
    }

    enableChat() {
        const messageInput = document.querySelector('#message-input');
        const submitBtn = document.querySelector('#chat-form button[type="submit"]');
        
        if (messageInput && submitBtn) {
            messageInput.disabled = false;
            submitBtn.disabled = false;
            messageInput.placeholder = "Ask about the video... (e.g., 'What happens at 30 seconds?')";
            
            // Add visual indication that chat is enabled
            messageInput.style.borderColor = 'var(--accent-color)';
            setTimeout(() => {
                messageInput.style.borderColor = '';
            }, 2000);
        }
    }

    updateVideoPlayer(videoFile) {
        const videoPlayer = document.querySelector('#video-player');
        if (videoPlayer && videoFile) {
            // Update video source to show the selected video
            const videoUrl = `/video-stream/${videoFile}`;
            videoPlayer.src = videoUrl;
            
            // Find the source element and update it too
            const sourceEl = videoPlayer.querySelector('source');
            if (sourceEl) {
                sourceEl.src = videoUrl;
            }
            
            // Reload the video
            videoPlayer.load();
            
            console.log(`Updated video player to show: ${videoFile}`);
        }
    }

    async sendMessage() {
        const messageInput = document.querySelector('#message-input');
        const message = messageInput.value.trim();
        
        if (!message) return;

        if (!this.currentVideo) {
            window.showNotification('Please process a video first', 'warning');
            return;
        }

        this.addMessageToChat('user', message);
        messageInput.value = '';

        const responseEl = this.addMessageToChat('assistant', '');
        this.showTypingIndicator(responseEl);
        
        let currentResponse = '';

        try {
            // Use simplified non-streaming approach for better performance
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    enable_temporal: document.querySelector('#enable-temporal')?.checked || false,
                    generate_snippets: document.querySelector('#generate-snippets')?.checked || false
                })
            });

            const data = await response.json();
            
            this.hideTypingIndicator(responseEl);
            
            if (data.success) {
                responseEl.innerHTML = this.formatResponse(data.response);
                
                // Show processing info in a compact way
                if (data.inference_time || data.frames_processed) {
                    const infoEl = document.createElement('div');
                    infoEl.className = 'mt-2';
                    infoEl.style.fontSize = '0.75rem';
                    infoEl.style.color = 'var(--text-secondary)';
                    infoEl.innerHTML = `⚡ ${data.inference_time?.toFixed(1)}s • ${data.frames_processed} frames`;
                    responseEl.appendChild(infoEl);
                }
                
                if (data.video_snippets && data.video_snippets.length > 0) {
                    this.addVideoSnippets(data.video_snippets);
                }
            } else {
                responseEl.innerHTML = `<div class="status-indicator status-error">Error: ${data.error}</div>`;
            }
        } catch (error) {
            this.hideTypingIndicator(responseEl);
            responseEl.innerHTML = `<div class="status-indicator status-error">Error: ${error.message}</div>`;
        }
    }

    addMessageToChat(type, content) {
        const chatMessages = document.querySelector('#chat-messages');
        if (!chatMessages) return null;

        const messageEl = document.createElement('div');
        messageEl.className = `chat-message ${type}`;
        messageEl.innerHTML = content;

        chatMessages.appendChild(messageEl);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return messageEl;
    }

    showTypingIndicator(element) {
        element.innerHTML = '<div class="typing-indicator"><i class="fas fa-spinner fa-spin"></i> Analyzing video...</div>';
    }

    hideTypingIndicator(element) {
        // This will be replaced with actual content
    }

    updateLoadingMessage(element, message) {
        element.innerHTML = `<div class="typing-indicator"><i class="fas fa-spinner fa-spin"></i> ${message}</div>`;
    }

    formatResponse(response) {
        // Convert markdown-like formatting to HTML
        return response
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    addVideoSnippets(snippets) {
        const chatMessages = document.querySelector('#chat-messages');
        if (!chatMessages || !snippets.length) return;

        const snippetsEl = document.createElement('div');
        snippetsEl.className = 'video-snippets mt-3';
        snippetsEl.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <div class="card-title">Related Video Segments</div>
                </div>
                <div class="card-body">
                    ${snippets.map(snippet => `
                        <div class="snippet-item mb-2">
                            <strong>${snippet.timestamp}s:</strong> ${snippet.description}
                            <button class="btn btn-outline btn-sm ml-2" onclick="playSnippet(${snippet.timestamp})">
                                Play
                            </button>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        chatMessages.appendChild(snippetsEl);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    clearChat() {
        const chatMessages = document.querySelector('#chat-messages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }

        fetch('/chat/clear', { method: 'POST' })
            .catch(error => console.error('Failed to clear chat:', error));
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
        const outputFormat = document.querySelector('input[name="output-format"]:checked');

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
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            this.handleRealTimeUpdate(data);
                        } catch (e) {
                            console.warn('Failed to parse SSE data:', line);
                        }
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
        const liveChunks = document.querySelector('#live-chunks');
        
        switch (data.type) {
            case 'status':
                progressText.textContent = data.message;
                break;
                
            case 'processing_info':
                progressText.textContent = `Processing ${data.total_chunks} chunks with ${data.total_frames} total frames`;
                break;
                
            case 'chunk_start':
                progressText.textContent = `Processing chunk ${data.chunk}/${data.total_chunks} (${data.start_time.toFixed(1)}s - ${data.end_time.toFixed(1)}s)`;
                break;
                
            case 'chunk_complete':
                // Update progress
                const progress = data.progress || 0;
                progressFill.style.width = `${progress}%`;
                progressText.textContent = `Completed chunk ${data.chunk}/${data.total_chunks} (${progress.toFixed(1)}%)`;
                
                // Add chunk result to live display
                const chunkDiv = document.createElement('div');
                chunkDiv.className = 'chunk-result';
                chunkDiv.innerHTML = `
                    <div class="chunk-header">
                        <strong>📹 Chunk ${data.chunk}/${data.total_chunks}</strong>
                        <span class="chunk-timing">${data.processing_time.toFixed(1)}s • ${data.characters} chars</span>
                    </div>
                    <div class="chunk-content">${data.caption}</div>
                `;
                liveChunks.appendChild(chunkDiv);
                liveChunks.scrollTop = liveChunks.scrollHeight; // Auto-scroll
                break;
                
            case 'cleanup':
                // Show cleanup notifications briefly
                const cleanupMsg = document.createElement('div');
                cleanupMsg.className = 'cleanup-msg';
                cleanupMsg.textContent = `🧹 ${data.message}`;
                cleanupMsg.style.opacity = '0.7';
                cleanupMsg.style.fontSize = '0.8rem';
                liveChunks.appendChild(cleanupMsg);
                setTimeout(() => cleanupMsg.remove(), 3000);
                break;
                
            case 'complete':
                // Show final combined caption
                const finalDiv = document.querySelector('#final-caption');
                finalDiv.innerHTML = `
                    <div class="final-caption-header">
                        <h3>✅ Final Caption</h3>
                        <div class="caption-stats">
                            ${data.caption.length} characters • ${data.chunks_processed} chunks • ${data.total_frames} frames processed
                        </div>
                    </div>
                    <div class="final-caption-content">${data.caption}</div>
                `;
                finalDiv.classList.remove('hidden');
                progressText.textContent = `✅ Completed! ${data.chunks_processed} chunks processed successfully`;
                progressFill.style.width = '100%';
                break;
                
            case 'error':
            case 'chunk_error':
                console.error('Processing error:', data);
                progressText.innerHTML = `<span style="color: red;">Error: ${data.message || data.error}</span>`;
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

        estimationEl.innerHTML = '<div class="spinner"></div> Calculating...';

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
                    <div><strong>📹 Video:</strong> ${duration}s (${Math.round(data.video_info.fps)} FPS)</div>
                    <div><strong>📊 Processing:</strong> ${chunks} chunks${modeDescription}</div>
                    <div><strong>🎯 Frames:</strong> ${totalFrames} total (~${Math.round(totalFrames/chunks)}/chunk)</div>
                    <div><strong>⏱️ Time:</strong> ~${data.time_estimation.total_time_minutes} minutes</div>
                    <div style="font-size: 0.7rem; opacity: 0.8; margin-top: 0.5rem;">
                        Processing: ${data.time_estimation.processing_time_seconds}s + Model loading: 16s
                    </div>
                `;

                // Show chunk preview if available
                if (data.chunk_details && data.chunk_details.length > 0) {
                    const previewHtml = data.chunk_details.slice(0, 3).map(chunk => 
                        `Chunk ${chunk.chunk_number}: ${chunk.start_time}s-${chunk.end_time}s (${chunk.frames} frames)`
                    ).join('<br>');
                    
                    estimationEl.innerHTML += `
                        <div style="font-size: 0.7rem; margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #e5e7eb;">
                            <strong>Preview:</strong><br>
                            ${previewHtml}
                            ${data.chunk_details.length > 3 ? '<br>...' : ''}
                        </div>
                    `;
                }
            } else {
                estimationEl.innerHTML = `<div style="color: var(--danger-color);">Error: ${data.error}</div>`;
            }
        } catch (error) {
            estimationEl.innerHTML = `<div style="color: var(--danger-color);">Failed to calculate estimate</div>`;
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

// Global functions
function playSnippet(timestamp) {
    // This would integrate with video player to seek to specific timestamp
    console.log('Playing snippet at:', timestamp);
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.SurveillanceApp = new SurveillanceApp();
    window.ChatInterface = new ChatInterface();
    window.ChatInterface.init();
    window.CaptionGenerator = new CaptionGenerator();
    window.CaptionGenerator.init();
});