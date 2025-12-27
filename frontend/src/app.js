// Paperlinse Frontend Application

// ============================================================================
// SSE (Server-Sent Events) Client for Real-time Updates
// ============================================================================

const SSE = {
    eventSource: null,
    reconnectAttempts: 0,
    maxReconnectAttempts: 10,
    baseReconnectDelay: 1000,
    maxReconnectDelay: 30000,
    reconnectTimer: null,
    isConnected: false,
    
    connect() {
        if (this.eventSource) {
            return;
        }
        
        console.log('SSE: Connecting to /api/events...');
        
        try {
            this.eventSource = new EventSource('/api/events');
            
            this.eventSource.onopen = () => {
                console.log('SSE: Connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
            };
            
            this.eventSource.onmessage = (event) => {
                this.handleMessage(event);
            };
            
            this.eventSource.onerror = (error) => {
                console.error('SSE: Connection error');
                this.isConnected = false;
                
                if (this.eventSource) {
                    this.eventSource.close();
                    this.eventSource = null;
                }
                
                this.scheduleReconnect();
            };
            
        } catch (error) {
            console.error('SSE: Failed to create EventSource', error);
            this.scheduleReconnect();
        }
    },
    
    disconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        
        this.isConnected = false;
        this.reconnectAttempts = 0;
    },
    
    scheduleReconnect() {
        if (this.reconnectTimer) return;
        
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('SSE: Max reconnection attempts reached');
            return;
        }
        
        const delay = Math.min(
            this.baseReconnectDelay * Math.pow(2, this.reconnectAttempts) + Math.random() * 1000,
            this.maxReconnectDelay
        );
        
        this.reconnectAttempts++;
        console.log(`SSE: Reconnecting in ${Math.round(delay / 1000)}s (attempt ${this.reconnectAttempts})`);
        
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, delay);
    },
    
    handleMessage(event) {
        try {
            const payload = JSON.parse(event.data);
            const { type, data } = payload;
            
            switch (type) {
                case 'processing_started':
                    console.log('SSE: Processing started', data.file_name);
                    // Refresh queue to show processing status
                    if (isPageActive('queue')) {
                        loadQueue();
                    }
                    break;
                    
                case 'processing_progress':
                    // Progress bar will update via its own polling, but we can update queue
                    // Only refresh every few progress updates to avoid excessive DOM updates
                    if (data.progress_percent % 20 === 0 && isPageActive('queue')) {
                        loadQueue();
                    }
                    break;
                    
                case 'processing_completed':
                    console.log('SSE: Processing completed', data.file_name);
                    showNotification(`Document processed: ${data.file_name}`, 'success');
                    refreshAfterProcessing();
                    break;
                    
                case 'processing_error':
                    console.log('SSE: Processing error', data.file_name, data.error);
                    showNotification(`Processing failed: ${data.file_name}`, 'error');
                    refreshAfterProcessing();
                    break;
                    
                case 'queue_updated':
                    console.log('SSE: Queue updated');
                    if (isPageActive('queue')) {
                        loadQueue();
                    }
                    break;
                    
                case 'document_approved':
                case 'document_rejected':
                    console.log('SSE: Document status changed', type);
                    refreshAfterDocumentChange();
                    break;
                    
                case 'stats_updated':
                    console.log('SSE: Stats updated');
                    loadStats();
                    break;
                    
                case 'scan_completed':
                    console.log('SSE: Scan completed, found', data.files_found, 'files');
                    if (isPageActive('queue')) {
                        loadQueue();
                    }
                    loadStats();
                    break;
                    
                case 'heartbeat':
                    // Connection alive, no action needed
                    break;
                    
                default:
                    console.log('SSE: Unknown event', type);
            }
            
        } catch (error) {
            console.error('SSE: Failed to parse message:', event.data, error);
        }
    }
};

// Helper to check if a page is currently active
function isPageActive(pageName) {
    const page = document.getElementById(`page-${pageName}`);
    return page && page.classList.contains('active');
}

// Helper to refresh UI after processing completes
function refreshAfterProcessing() {
    loadStats();
    if (isPageActive('queue')) {
        loadQueue();
    }
    if (isPageActive('documents')) {
        loadDocuments();
    }
}

// Helper to refresh UI after document status changes
function refreshAfterDocumentChange() {
    loadStats();
    if (isPageActive('queue')) {
        loadQueue();
    }
    if (isPageActive('documents')) {
        loadDocuments();
    }
}

// ============================================================================
// Notification System
// ============================================================================

function showNotification(message, type = 'info', duration = 4000) {
    const container = document.getElementById('notification-container');
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    const icons = {
        success: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
        error: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>',
        info: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
        warning: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
    };
    
    notification.innerHTML = `
        <span class="notification-icon">${icons[type] || icons.info}</span>
        <div class="notification-content">
            <span class="notification-message">${escapeHtml(message)}</span>
        </div>
        <button class="notification-close" onclick="this.parentElement.remove()">&times;</button>
    `;
    
    container.appendChild(notification);
    
    // Auto-remove after duration
    if (duration > 0) {
        setTimeout(() => {
            notification.classList.add('hiding');
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }
    
    return notification;
}

// ============================================================================
// Rejection Modal
// ============================================================================

let pendingRejectionDocId = null;

function openRejectionModal(docId) {
    pendingRejectionDocId = docId;
    document.getElementById('rejection-reason').value = '';
    document.getElementById('rejection-modal').classList.add('active');
    document.getElementById('rejection-reason').focus();
}

function closeRejectionModal() {
    document.getElementById('rejection-modal').classList.remove('active');
    pendingRejectionDocId = null;
}

async function confirmRejection() {
    if (!pendingRejectionDocId) return;
    
    const reason = document.getElementById('rejection-reason').value.trim();
    const docId = pendingRejectionDocId;
    
    closeRejectionModal();
    
    try {
        await apiPost(`/documents/${docId}/reject?reason=${encodeURIComponent(reason)}`);
        showNotification('Document rejected', 'success');
        loadStats();
        loadQueue();
        loadDocuments();
    } catch (error) {
        console.error('Failed to reject document:', error);
        showNotification('Failed to reject document', 'error');
    }
}

// ============================================================================
// Navigation
// ============================================================================

document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        const page = item.dataset.page;
        navigateTo(page);
    });
});

function navigateTo(page) {
    // Update nav
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    document.querySelector(`.nav-item[data-page="${page}"]`).classList.add('active');
    
    // Update page
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${page}`).classList.add('active');
    
    // Load page-specific data
    if (page === 'documents') {
        loadDocuments();
    } else if (page === 'queue') {
        loadQueue();
    } else if (page === 'dashboard') {
        loadStats();
    } else if (page === 'logs') {
        loadLogs();
    }
}

// ============================================================================
// API calls
// ============================================================================

async function apiGet(endpoint) {
    const response = await fetch(`/api${endpoint}`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

async function apiPost(endpoint, data) {
    const response = await fetch(`/api${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

async function apiPut(endpoint, data) {
    const response = await fetch(`/api${endpoint}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

async function apiDelete(endpoint) {
    const response = await fetch(`/api${endpoint}`, {
        method: 'DELETE'
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

// ============================================================================
// Configuration
// ============================================================================

// Store current config globally for edit/cancel operations
let currentConfig = null;

async function loadConfig() {
    try {
        const config = await apiGet('/config');
        currentConfig = config;
        
        // Populate incoming share fields
        document.getElementById('incoming-path').value = config.incoming_share.path || '';
        document.getElementById('poll-interval').value = config.incoming_share.poll_interval_seconds || 30;
        document.getElementById('incoming-username').value = config.incoming_share.credentials.username || '';
        document.getElementById('incoming-password').value = config.incoming_share.credentials.password || '';
        document.getElementById('incoming-domain').value = config.incoming_share.credentials.domain || '';
        
        // Populate storage fields
        document.getElementById('storage-path').value = config.storage.path || '';
        document.getElementById('storage-username').value = config.storage.credentials.username || '';
        document.getElementById('storage-password').value = config.storage.credentials.password || '';
        document.getElementById('storage-domain').value = config.storage.credentials.domain || '';
        
        // Update summary views
        updateShareSummary('incoming', config.incoming_share);
        updateShareSummary('storage', config.storage);
        
        // Show appropriate view based on config state
        const incomingConfigured = !!config.incoming_share.path;
        const storageConfigured = !!config.storage.path;
        
        showShareView('incoming', incomingConfigured ? 'summary' : 'form');
        showShareView('storage', storageConfigured ? 'summary' : 'form');
        
        // Update dashboard based on config
        updateDashboard(config);
        
        // Load processing mode for settings page
        loadSettingsProcessingMode();
        
        // Load LLM config for settings page
        loadLLMConfig();
        
        // Load processing config for settings page
        loadProcessingConfig();
        
        // Load Vision LLM config for settings page
        loadVisionLLMConfig();
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

function updateShareSummary(type, shareConfig) {
    const pathEl = document.getElementById(`${type}-summary-path`);
    const statusEl = document.getElementById(`${type}-summary-status`);
    const credsEl = document.getElementById(`${type}-summary-creds`);
    const credsRowEl = document.getElementById(`${type}-summary-creds-row`);
    
    if (shareConfig.path) {
        pathEl.textContent = shareConfig.path;
        statusEl.innerHTML = '<span class="status-indicator status-pending">Checking...</span>';
        
        // Check connection status
        testConnectionSilent(type, shareConfig);
        
        // Show credentials info
        if (shareConfig.credentials && shareConfig.credentials.username) {
            const domain = shareConfig.credentials.domain ? `${shareConfig.credentials.domain}\\` : '';
            credsEl.textContent = `${domain}${shareConfig.credentials.username}`;
            credsRowEl.style.display = '';
        } else {
            credsRowEl.style.display = 'none';
        }
    } else {
        pathEl.textContent = 'Not configured';
        statusEl.textContent = '-';
        credsRowEl.style.display = 'none';
    }
}

async function testConnectionSilent(type, shareConfig) {
    const statusEl = document.getElementById(`${type}-summary-status`);
    
    try {
        const result = await apiPost('/test-connection', {
            path: shareConfig.path,
            credentials: shareConfig.credentials || { username: '', password: '', domain: '' }
        });
        
        if (result.success) {
            statusEl.innerHTML = `<span class="status-indicator status-success">Connected${result.writable ? ' (writable)' : ' (read-only)'}</span>`;
        } else {
            statusEl.innerHTML = `<span class="status-indicator status-error">Error: ${result.message}</span>`;
        }
    } catch (error) {
        statusEl.innerHTML = '<span class="status-indicator status-error">Connection failed</span>';
    }
}

function showShareView(type, view) {
    const summaryEl = document.getElementById(`${type}-summary`);
    const formEl = document.getElementById(`${type}-form`);
    const editBtn = document.getElementById(`${type}-edit-btn`);
    
    if (view === 'summary') {
        summaryEl.style.display = '';
        formEl.style.display = 'none';
        editBtn.textContent = 'Edit';
    } else {
        summaryEl.style.display = 'none';
        formEl.style.display = '';
        editBtn.textContent = 'Cancel';
    }
}

function toggleShareEdit(type) {
    const formEl = document.getElementById(`${type}-form`);
    const isEditing = formEl.style.display !== 'none';
    
    if (isEditing) {
        // Cancel edit - restore original values
        cancelShareEdit(type);
    } else {
        // Start editing
        showShareView(type, 'form');
    }
}

function cancelShareEdit(type) {
    // Restore original values from currentConfig
    if (currentConfig) {
        const shareConfig = type === 'incoming' ? currentConfig.incoming_share : currentConfig.storage;
        document.getElementById(`${type}-path`).value = shareConfig.path || '';
        document.getElementById(`${type}-username`).value = shareConfig.credentials.username || '';
        document.getElementById(`${type}-password`).value = shareConfig.credentials.password || '';
        document.getElementById(`${type}-domain`).value = shareConfig.credentials.domain || '';
    }
    
    // Only show summary if there's a configured path
    const pathValue = currentConfig ? 
        (type === 'incoming' ? currentConfig.incoming_share.path : currentConfig.storage.path) : '';
    
    if (pathValue) {
        showShareView(type, 'summary');
    }
    
    // Clear test result
    document.getElementById(`${type}-test-result`).style.display = 'none';
}

async function saveShareConfig(type) {
    const statusEl = document.getElementById(`${type}-save-status`);
    statusEl.className = 'save-status';
    statusEl.textContent = 'Saving...';
    
    // Build the config update
    const shareConfig = {
        path: document.getElementById(`${type}-path`).value,
        credentials: {
            username: document.getElementById(`${type}-username`).value,
            password: document.getElementById(`${type}-password`).value,
            domain: document.getElementById(`${type}-domain`).value
        }
    };
    
    // Include poll_interval for incoming share
    if (type === 'incoming') {
        shareConfig.poll_interval_seconds = parseInt(document.getElementById('poll-interval').value) || 30;
    }
    
    // Build full config preserving the other share's settings
    const config = {
        incoming_share: type === 'incoming' ? shareConfig : {
            path: document.getElementById('incoming-path').value,
            poll_interval_seconds: parseInt(document.getElementById('poll-interval').value) || 30,
            credentials: {
                username: document.getElementById('incoming-username').value,
                password: document.getElementById('incoming-password').value,
                domain: document.getElementById('incoming-domain').value
            }
        },
        storage: type === 'storage' ? shareConfig : {
            path: document.getElementById('storage-path').value,
            credentials: {
                username: document.getElementById('storage-username').value,
                password: document.getElementById('storage-password').value,
                domain: document.getElementById('storage-domain').value
            }
        }
    };
    
    try {
        await apiPost('/config', config);
        statusEl.className = 'save-status success';
        statusEl.textContent = 'Saved!';
        setTimeout(() => { statusEl.textContent = ''; }, 2000);
        
        // Reload config to update summaries
        await loadConfig();
    } catch (error) {
        statusEl.className = 'save-status error';
        statusEl.textContent = 'Failed to save: ' + error.message;
    }
}

async function saveSchedulerConfig() {
    const statusEl = document.getElementById('scheduler-save-status');
    statusEl.className = 'save-status';
    statusEl.textContent = 'Saving...';
    
    const pollInterval = parseInt(document.getElementById('poll-interval').value) || 30;
    
    // Build config with updated poll interval
    const config = {
        incoming_share: {
            path: document.getElementById('incoming-path').value,
            poll_interval_seconds: pollInterval,
            credentials: {
                username: document.getElementById('incoming-username').value,
                password: document.getElementById('incoming-password').value,
                domain: document.getElementById('incoming-domain').value
            }
        },
        storage: {
            path: document.getElementById('storage-path').value,
            credentials: {
                username: document.getElementById('storage-username').value,
                password: document.getElementById('storage-password').value,
                domain: document.getElementById('storage-domain').value
            }
        }
    };
    
    try {
        await apiPost('/config', config);
        statusEl.className = 'save-status success';
        statusEl.textContent = 'Saved!';
        setTimeout(() => { statusEl.textContent = ''; }, 2000);
        
        currentConfig = config;
    } catch (error) {
        statusEl.className = 'save-status error';
        statusEl.textContent = 'Failed to save: ' + error.message;
    }
}

async function loadSettingsProcessingMode() {
    try {
        const response = await apiGet('/processing/mode');
        const isAutomatic = response.mode === 'automatic';
        
        document.getElementById('mode-automatic').checked = isAutomatic;
        document.getElementById('mode-manual').checked = !isAutomatic;
    } catch (error) {
        console.error('Failed to load processing mode:', error);
    }
}

async function saveProcessingMode() {
    const statusEl = document.getElementById('mode-save-status');
    statusEl.className = 'save-status';
    statusEl.textContent = 'Saving...';
    
    const isAutomatic = document.getElementById('mode-automatic').checked;
    const newMode = isAutomatic ? 'automatic' : 'manual';
    
    try {
        await apiPut('/processing/mode', { mode: newMode });
        statusEl.className = 'save-status success';
        statusEl.textContent = 'Saved!';
        setTimeout(() => { statusEl.textContent = ''; }, 2000);
        
        // Also update the dashboard toggle if visible
        const dashboardToggle = document.getElementById('auto-mode-toggle');
        if (dashboardToggle) {
            dashboardToggle.checked = isAutomatic;
            document.getElementById('mode-label').textContent = isAutomatic ? 'Automatic' : 'Manual';
            document.getElementById('mode-description').textContent = isAutomatic 
                ? 'Documents are automatically processed and moved to storage.'
                : 'Documents require manual approval before being moved to storage.';
        }
    } catch (error) {
        statusEl.className = 'save-status error';
        statusEl.textContent = 'Failed to save: ' + error.message;
    }
}

// ============================================================================
// Processing Configuration
// ============================================================================

async function loadProcessingConfig() {
    try {
        const config = await apiGet('/config/processing');
        document.getElementById('concurrent-workers').value = config.concurrent_workers || 3;
        document.getElementById('batch-size').value = config.batch_size || 10;
    } catch (error) {
        console.error('Failed to load processing config:', error);
    }
}

async function saveProcessingConfig() {
    const statusEl = document.getElementById('processing-config-save-status');
    statusEl.className = 'save-status';
    statusEl.textContent = 'Saving...';
    
    const concurrentWorkers = parseInt(document.getElementById('concurrent-workers').value) || 3;
    const batchSize = parseInt(document.getElementById('batch-size').value) || 10;
    
    try {
        await apiPost('/config/processing', {
            concurrent_workers: concurrentWorkers,
            batch_size: batchSize
        });
        
        statusEl.className = 'save-status success';
        statusEl.textContent = 'Saved!';
        showNotification('Processing settings saved', 'success');
        setTimeout(() => { statusEl.textContent = ''; }, 2000);
    } catch (error) {
        statusEl.className = 'save-status error';
        statusEl.textContent = 'Failed to save: ' + error.message;
        showNotification('Failed to save processing config: ' + error.message, 'error');
    }
}

// ============================================================================
// Progress Bar
// ============================================================================

let progressPollInterval = null;

function startProgressPolling() {
    // Poll immediately, then every 2 seconds
    updateProgressFromServer();
    progressPollInterval = setInterval(updateProgressFromServer, 2000);
}

function stopProgressPolling() {
    if (progressPollInterval) {
        clearInterval(progressPollInterval);
        progressPollInterval = null;
    }
}

async function updateProgressFromServer() {
    try {
        const data = await apiGet('/processing/progress');
        updateProgressBar(data);
    } catch (error) {
        // Silently fail - server may be restarting
        console.debug('Progress poll failed:', error);
    }
}

async function cancelProcessing(filePath = null) {
    try {
        const cancelBtn = document.getElementById('progress-cancel-btn');
        if (cancelBtn) {
            cancelBtn.disabled = true;
            cancelBtn.style.opacity = '0.5';
        }
        
        const response = await apiPost('/processing/cancel', { file_path: filePath });
        
        if (response.success) {
            if (response.cancelled_count > 0) {
                showNotification(`Cancelled ${response.cancelled_count} file(s)`, 'info');
            } else {
                showNotification('No files were being processed', 'info');
            }
        } else {
            showNotification(`Failed to cancel: ${response.message}`, 'error');
        }
        
        // Refresh progress bar and queue page
        await updateProgressFromServer();
        if (isPageActive('queue')) {
            await loadIncomingFiles();
        }
        
    } catch (error) {
        console.error('Error cancelling processing:', error);
        showNotification('Failed to cancel processing', 'error');
    } finally {
        const cancelBtn = document.getElementById('progress-cancel-btn');
        if (cancelBtn) {
            cancelBtn.disabled = false;
            cancelBtn.style.opacity = '1';
        }
    }
}

function updateProgressBar(data) {
    const container = document.getElementById('progress-bar-container');
    const statusEl = document.getElementById('progress-status');
    const detailsEl = document.getElementById('progress-details');
    const fillEl = document.getElementById('progress-fill');
    const percentageEl = document.getElementById('progress-percentage');
    const stepsEl = document.getElementById('progress-steps');
    
    if (!container) return;
    
    // Show/hide based on active processing
    if (data.is_active && data.total > 0) {
        container.classList.remove('hidden');
        container.classList.add('active');
        
        // Calculate percentage
        const completed = data.completed + data.error;
        const percentage = Math.round((completed / data.total) * 100);
        
        // Update UI
        statusEl.textContent = `Processing ${data.processing} document${data.processing !== 1 ? 's' : ''}...`;
        detailsEl.textContent = `${completed}/${data.total} completed${data.queued > 0 ? `, ${data.queued} queued` : ''}`;
        fillEl.style.width = `${percentage}%`;
        percentageEl.textContent = `${percentage}%`;
        
        // Show detailed processing steps
        if (stepsEl && data.processing_details && data.processing_details.length > 0) {
            const llmModel = data.llm_model || '';
            const stepsHtml = data.processing_details.map(detail => {
                const stepIcon = getStepIcon(detail.step);
                const stepName = getStepName(detail.step, llmModel);
                return `
                    <div class="progress-step-item">
                        <span class="step-icon">${stepIcon}</span>
                        <span class="step-file" title="${escapeAttr(detail.file_name)}">${escapeHtml(truncateText(detail.file_name, 25))}</span>
                        <span class="step-name">${stepName}</span>
                        <span class="step-detail">${escapeHtml(detail.step_detail)}</span>
                        <span class="step-progress">${detail.progress_percent}%</span>
                    </div>
                `;
            }).join('');
            stepsEl.innerHTML = stepsHtml;
            stepsEl.style.display = 'block';
        } else if (stepsEl) {
            stepsEl.style.display = 'none';
        }
        
        // Show current files being processed in tooltip
        if (data.current_files && data.current_files.length > 0) {
            const fileNames = data.current_files.map(f => f.split('/').pop()).join(', ');
            statusEl.title = `Processing: ${fileNames}`;
        }
    } else {
        container.classList.add('hidden');
        container.classList.remove('active');
        if (stepsEl) stepsEl.style.display = 'none';
    }
}

function getStepIcon(step) {
    const icons = {
        'queued': '⏳',
        'starting': '🚀',
        'ocr': '📝',
        'llm': '🤖',
        'saving': '💾',
        'completed': '✅',
        'cancelled': '🚫',
        'error': '❌'
    };
    return icons[step] || '⏳';
}

function getStepName(step, llmModel = '') {
    const names = {
        'queued': 'Queued',
        'starting': 'Starting',
        'ocr': 'OCR',
        'llm': llmModel ? `Checking text (${llmModel})` : 'Checking text',
        'saving': 'Saving',
        'completed': 'Done',
        'cancelled': 'Cancelled',
        'error': 'Error'
    };
    return names[step] || step;
}

// ============================================================================
// LLM Configuration and Model Management
// ============================================================================

let currentPullEventSource = null;

async function loadLLMConfig() {
    try {
        // Load prompts config
        const config = await apiGet('/config/llm');
        document.getElementById('llm-system-prompt').value = config.system_prompt || '';
        document.getElementById('llm-user-prompt-de').value = config.user_prompt_de || '';
        document.getElementById('llm-user-prompt-en').value = config.user_prompt_en || '';
        
        // Load models list
        await loadModels();
    } catch (error) {
        console.error('Failed to load LLM config:', error);
    }
}

async function loadModels() {
    const loadingEl = document.getElementById('models-loading');
    const listEl = document.getElementById('models-list');
    const statusContainer = document.getElementById('models-status-container');
    
    if (loadingEl) loadingEl.style.display = 'block';
    
    try {
        const data = await apiGet('/models');
        
        if (loadingEl) loadingEl.style.display = 'none';
        
        // Check if we have any vision models active
        const activeVisionModel = data.models.find(m => m.model_type === 'vision' && m.current);
        
        // Show Ollama status
        statusContainer.innerHTML = `
            <div class="ollama-status ${data.ollama_available ? 'connected' : 'disconnected'}">
                <span class="status-dot"></span>
                <span>Ollama ${data.ollama_available ? 'Connected' : 'Not Available'}</span>
            </div>
            ${activeVisionModel ? `
                <div class="vision-status connected">
                    <span class="status-dot"></span>
                    <span>Vision LLM Active</span>
                </div>
            ` : ''}
        `;
        
        if (!data.ollama_available && !data.models.some(m => m.model_type === 'vision' && m.installed)) {
            listEl.innerHTML = '<p class="placeholder-text">Cannot connect to Ollama and no vision models installed. Make sure Ollama is running or install a vision model.</p>';
            return;
        }
        
        // Separate text and vision models for better organization
        const textModels = data.models.filter(m => m.model_type !== 'vision');
        const visionModels = data.models.filter(m => m.model_type === 'vision');
        
        let modelsHtml = '';
        
        // Render text models
        if (textModels.length > 0) {
            modelsHtml += '<h4 class="models-section-title">Text Models (Ollama)</h4>';
            modelsHtml += textModels.map(model => renderModelCard(model)).join('');
        }
        
        // Render vision models
        if (visionModels.length > 0) {
            modelsHtml += '<h4 class="models-section-title">Vision Models (OpenVINO)</h4>';
            modelsHtml += '<p class="models-section-desc">Vision models extract metadata directly from document images, bypassing OCR for metadata extraction.</p>';
            modelsHtml += visionModels.map(model => renderModelCard(model)).join('');
        }
        
        listEl.innerHTML = modelsHtml || '<p class="placeholder-text">No models configured</p>';
        
    } catch (error) {
        console.error('Failed to load models:', error);
        if (loadingEl) loadingEl.style.display = 'none';
        statusContainer.innerHTML = '<span class="input-status error">Error loading models</span>';
    }
}

function renderModelCard(model) {
    const installedIcon = model.installed 
        ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
        : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>';
    
    // Badge for status
    let badge = '';
    if (model.current) {
        badge = '<span class="model-badge current">Active</span>';
    } else if (model.pulling) {
        badge = '<span class="model-badge pulling">Downloading...</span>';
    }
    
    // Badge for model type
    let typeBadge = '';
    if (model.model_type === 'vision') {
        typeBadge = '<span class="model-badge vision">Vision</span>';
    }
    
    let actions = '';
    if (model.pulling) {
        actions = `<span class="loading-text">${model.pull_progress}%</span>`;
    } else if (model.installed) {
        if (!model.current) {
            actions = `
                <button class="btn btn-primary btn-small" onclick="switchModel('${escapeAttr(model.id)}')">Use</button>
                <button class="btn btn-secondary btn-small" onclick="deleteModel('${escapeAttr(model.id)}')" title="Delete">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                </button>
            `;
        } else {
            actions = '<span class="input-status success">In Use</span>';
        }
    } else {
        actions = `<button class="btn btn-primary btn-small" onclick="downloadModel('${escapeAttr(model.id)}')">Download</button>`;
    }
    
    // Size display - vision models are in MB but often larger, keep same format
    const sizeDisplay = model.size_mb >= 1000 
        ? `~${(model.size_mb / 1024).toFixed(1)} GB` 
        : `~${model.size_mb} MB`;
    
    return `
        <div class="model-card ${model.current ? 'current' : ''} ${!model.installed ? 'not-installed' : ''} ${model.model_type === 'vision' ? 'vision-model' : ''}">
            <div class="model-status-icon ${model.installed ? 'installed' : 'not-installed'}">
                ${installedIcon}
            </div>
            <div class="model-info">
                <div class="model-name">
                    ${escapeHtml(model.name)}
                    <span class="model-id">${escapeHtml(model.id)}</span>
                    ${typeBadge}
                    ${badge}
                </div>
                <div class="model-meta">
                    <span class="size">${sizeDisplay}</span>
                    ${model.model_type === 'vision' ? '<span class="model-type-label">OpenVINO</span>' : '<span class="model-type-label">Ollama</span>'}
                </div>
                <div class="model-description">${escapeHtml(model.description)}</div>
            </div>
            <div class="model-actions">
                ${actions}
            </div>
        </div>
    `;
}

async function switchModel(modelId) {
    try {
        showNotification(`Switching to ${modelId}...`, 'info');
        
        const result = await apiPost(`/models/${encodeURIComponent(modelId)}/switch`);
        
        if (result.status === 'download_required') {
            // Model needs to be downloaded first - show notification and offer download
            const modelType = result.model_type === 'vision' ? 'Vision model' : 'Model';
            showNotification(`${modelType} ${modelId} is not installed. Click Download to install it.`, 'warning', 6000);
            return;
        }
        
        // Show appropriate message based on model type
        if (result.model_type === 'vision') {
            showNotification(`Switched to vision model ${modelId}. Vision LLM is now enabled.`, 'success');
        } else {
            showNotification(`Switched to ${modelId}`, 'success');
        }
        
        await loadModels();
    } catch (error) {
        console.error('Failed to switch model:', error);
        showNotification('Failed to switch model: ' + error.message, 'error');
    }
}

async function downloadModel(modelId) {
    const progressEl = document.getElementById('model-pull-progress');
    const modelNameEl = document.getElementById('pull-model-name');
    const statusEl = document.getElementById('pull-status');
    const fillEl = document.getElementById('pull-progress-fill');
    const percentEl = document.getElementById('pull-percentage');
    
    // Show progress UI
    progressEl.style.display = 'block';
    modelNameEl.textContent = `Downloading ${modelId}...`;
    statusEl.textContent = 'Starting...';
    fillEl.style.width = '0%';
    percentEl.textContent = '0%';
    
    // Close any existing event source
    if (currentPullEventSource) {
        currentPullEventSource.close();
    }
    
    try {
        // Use EventSource for SSE
        const url = `/api/models/${encodeURIComponent(modelId)}/pull`;
        
        // We need to use fetch with streaming for POST requests
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let lastError = null;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Process SSE lines
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        // Update UI with message if available
                        if (data.message) {
                            statusEl.textContent = data.message;
                        } else if (data.status) {
                            statusEl.textContent = data.status;
                        }
                        
                        if (data.percent !== undefined) {
                            fillEl.style.width = `${data.percent}%`;
                            percentEl.textContent = `${data.percent}%`;
                        }
                        
                        if (data.status === 'success') {
                            showNotification(`Model ${modelId} downloaded successfully`, 'success');
                            progressEl.style.display = 'none';
                            await loadModels();
                            return;
                        }
                        
                        if (data.status === 'error') {
                            lastError = data.error || 'Download failed';
                            // Don't throw yet - let the stream finish
                        }
                    } catch (parseError) {
                        console.debug('SSE parse error:', parseError, 'line:', line);
                    }
                }
            }
        }
        
        // Check if we had an error during streaming
        if (lastError) {
            throw new Error(lastError);
        }
        
        // If we get here without success or error, reload models anyway
        await loadModels();
        progressEl.style.display = 'none';
        
    } catch (error) {
        console.error('Failed to download model:', error);
        showNotification('Failed to download model: ' + error.message, 'error');
        progressEl.style.display = 'none';
        await loadModels();
    }
}

async function deleteModel(modelId) {
    if (!confirm(`Delete model ${modelId}? This cannot be undone.`)) {
        return;
    }
    
    try {
        await apiDelete(`/models/${encodeURIComponent(modelId)}`);
        showNotification(`Model ${modelId} deleted`, 'success');
        await loadModels();
    } catch (error) {
        console.error('Failed to delete model:', error);
        showNotification('Failed to delete model: ' + error.message, 'error');
    }
}

async function saveLLMConfig() {
    const statusEl = document.getElementById('llm-save-status');
    statusEl.className = 'save-status';
    statusEl.textContent = 'Saving...';
    
    try {
        await apiPost('/config/llm', {
            system_prompt: document.getElementById('llm-system-prompt').value,
            user_prompt_de: document.getElementById('llm-user-prompt-de').value,
            user_prompt_en: document.getElementById('llm-user-prompt-en').value
        });
        
        statusEl.className = 'save-status success';
        statusEl.textContent = 'Saved!';
        showNotification('LLM prompts saved successfully', 'success');
        setTimeout(() => { statusEl.textContent = ''; }, 2000);
    } catch (error) {
        statusEl.className = 'save-status error';
        statusEl.textContent = 'Failed to save: ' + error.message;
        showNotification('Failed to save LLM config: ' + error.message, 'error');
    }
}

async function resetLLMConfig() {
    if (!confirm('Reset all LLM prompts to their default values?')) {
        return;
    }
    
    try {
        await apiPost('/config/llm/reset');
        await loadLLMConfig();
        showNotification('LLM prompts reset to defaults', 'success');
    } catch (error) {
        console.error('Failed to reset LLM config:', error);
        showNotification('Failed to reset LLM config: ' + error.message, 'error');
    }
}

// ============================================================================
// Vision LLM Configuration
// ============================================================================

async function loadVisionLLMConfig() {
    try {
        const config = await apiGet('/config/vision-llm');
        
        // Only update device selector - model selection is now handled via the main models list
        const deviceSelect = document.getElementById('vision-llm-device');
        const statusEl = document.getElementById('vision-llm-status');
        const maxPixelsSlider = document.getElementById('vision-llm-max-pixels');
        const maxPixelsDisplay = document.getElementById('vision-llm-max-pixels-display');
        const promptDeEl = document.getElementById('vision-llm-prompt-de');
        const promptEnEl = document.getElementById('vision-llm-prompt-en');
        
        if (deviceSelect) deviceSelect.value = config.device || 'CPU';
        
        // Set max_pixels slider value
        if (maxPixelsSlider) {
            maxPixelsSlider.value = config.max_pixels || 401408;
            updateMaxPixelsDisplay(maxPixelsSlider.value);
        }
        
        // Set vision prompts
        if (promptDeEl) promptDeEl.value = config.prompt_de || '';
        if (promptEnEl) promptEnEl.value = config.prompt_en || '';
        
        // Update status indicator based on whether vision LLM is active
        if (statusEl) {
            if (config.available) {
                statusEl.className = 'status-indicator status-success';
                statusEl.textContent = 'Active';
            } else if (config.enabled) {
                statusEl.className = 'status-indicator status-warning';
                statusEl.textContent = 'Not Ready';
                statusEl.title = config.status_message || 'Model not loaded';
            } else {
                statusEl.className = 'status-indicator status-disabled';
                statusEl.textContent = 'Disabled';
            }
        }
        
    } catch (error) {
        console.error('Failed to load Vision LLM config:', error);
    }
}

function updateMaxPixelsDisplay(value) {
    const display = document.getElementById('vision-llm-max-pixels-display');
    if (!display) return;
    
    const pixels = parseInt(value);
    if (pixels < 500000) {
        display.textContent = `~${Math.round(pixels / 1000)}k px (fast)`;
    } else if (pixels < 1500000) {
        display.textContent = `~${(pixels / 1000000).toFixed(1)}M px (balanced)`;
    } else {
        display.textContent = `~${(pixels / 1000000).toFixed(1)}M px (quality)`;
    }
}

async function saveVisionLLMConfig() {
    const statusEl = document.getElementById('vision-llm-save-status');
    statusEl.className = 'save-status';
    statusEl.textContent = 'Saving...';
    
    // Save device, max_pixels and prompt settings
    const device = document.getElementById('vision-llm-device').value;
    const maxPixelsSlider = document.getElementById('vision-llm-max-pixels');
    const maxPixels = maxPixelsSlider ? parseInt(maxPixelsSlider.value) : 401408;
    const promptDe = document.getElementById('vision-llm-prompt-de')?.value || '';
    const promptEn = document.getElementById('vision-llm-prompt-en')?.value || '';
    
    try {
        await apiPost('/config/vision-llm', {
            device: device,
            max_pixels: maxPixels,
            prompt_de: promptDe,
            prompt_en: promptEn
        });
        
        statusEl.className = 'save-status success';
        statusEl.textContent = 'Saved!';
        showNotification('Vision LLM settings saved', 'success');
        setTimeout(() => { statusEl.textContent = ''; }, 2000);
        
        // Reload to update status
        loadVisionLLMConfig();
    } catch (error) {
        statusEl.className = 'save-status error';
        statusEl.textContent = 'Failed to save: ' + error.message;
        showNotification('Failed to save Vision LLM config: ' + error.message, 'error');
    }
}

async function resetVisionLLMPrompts() {
    if (!confirm('Reset Vision LLM prompts to their default values?')) {
        return;
    }
    
    try {
        await apiPost('/config/vision-llm/reset-prompts');
        await loadVisionLLMConfig();
        showNotification('Vision LLM prompts reset to defaults', 'success');
    } catch (error) {
        console.error('Failed to reset Vision LLM prompts:', error);
        showNotification('Failed to reset Vision LLM prompts: ' + error.message, 'error');
    }
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Command copied to clipboard', 'success');
    }).catch(err => {
        console.error('Failed to copy:', err);
        showNotification('Failed to copy to clipboard', 'error');
    });
}

function updateDashboard(config) {
    const setupPrompt = document.getElementById('setup-prompt');
    const hasConfig = config.incoming_share.path && config.storage.path;
    
    if (hasConfig) {
        setupPrompt.style.display = 'none';
        loadStats();
        loadProcessingMode();
    } else {
        setupPrompt.style.display = 'block';
        document.getElementById('pending-count').textContent = '0';
        document.getElementById('awaiting-count').textContent = '0';
        document.getElementById('approved-count').textContent = '0';
        document.getElementById('error-count').textContent = '0';
    }
}

// ============================================================================
// Dashboard Statistics
// ============================================================================

async function loadStats() {
    try {
        const stats = await apiGet('/stats');
        document.getElementById('pending-count').textContent = stats.pending;
        document.getElementById('awaiting-count').textContent = stats.awaiting_approval;
        document.getElementById('approved-count').textContent = stats.approved;
        document.getElementById('error-count').textContent = stats.error;
    } catch (error) {
        console.error('Failed to load stats:', error);
        document.getElementById('pending-count').textContent = '?';
        document.getElementById('awaiting-count').textContent = '?';
        document.getElementById('approved-count').textContent = '?';
        document.getElementById('error-count').textContent = '?';
    }
}

// ============================================================================
// Processing Mode
// ============================================================================

async function loadProcessingMode() {
    try {
        const response = await apiGet('/processing/mode');
        const isAutomatic = response.mode === 'automatic';
        
        document.getElementById('auto-mode-toggle').checked = isAutomatic;
        document.getElementById('mode-label').textContent = isAutomatic ? 'Automatic' : 'Manual';
        document.getElementById('mode-description').textContent = isAutomatic 
            ? 'Documents are automatically processed and moved to storage.'
            : 'Documents require manual approval before being moved to storage.';
    } catch (error) {
        console.error('Failed to load processing mode:', error);
    }
}

async function toggleProcessingMode() {
    const toggle = document.getElementById('auto-mode-toggle');
    const newMode = toggle.checked ? 'automatic' : 'manual';
    
    try {
        await apiPut('/processing/mode', { mode: newMode });
        document.getElementById('mode-label').textContent = toggle.checked ? 'Automatic' : 'Manual';
        document.getElementById('mode-description').textContent = toggle.checked
            ? 'Documents are automatically processed and moved to storage.'
            : 'Documents require manual approval before being moved to storage.';
    } catch (error) {
        console.error('Failed to update processing mode:', error);
        toggle.checked = !toggle.checked; // Revert
    }
}

// ============================================================================
// Connection Testing
// ============================================================================

async function testConnection(type) {
    const resultEl = document.getElementById(`${type}-test-result`);
    const pathEl = document.getElementById(`${type === 'incoming' ? 'incoming' : 'storage'}-path`);
    const usernameEl = document.getElementById(`${type === 'incoming' ? 'incoming' : 'storage'}-username`);
    const passwordEl = document.getElementById(`${type === 'incoming' ? 'incoming' : 'storage'}-password`);
    const domainEl = document.getElementById(`${type === 'incoming' ? 'incoming' : 'storage'}-domain`);
    
    const path = pathEl.value.trim();
    if (!path) {
        resultEl.className = 'test-result error';
        resultEl.textContent = 'Please enter a path first';
        return;
    }
    
    resultEl.className = 'test-result';
    resultEl.style.display = 'block';
    resultEl.textContent = 'Testing connection...';
    
    try {
        const result = await apiPost('/test-connection', {
            path: path,
            credentials: {
                username: usernameEl.value,
                password: passwordEl.value,
                domain: domainEl.value
            }
        });
        
        if (result.success) {
            resultEl.className = 'test-result success';
            resultEl.textContent = result.message + (result.writable ? ' (writable)' : ' (read-only)');
        } else {
            resultEl.className = 'test-result error';
            resultEl.textContent = result.message;
        }
    } catch (error) {
        resultEl.className = 'test-result error';
        resultEl.textContent = 'Connection test failed: ' + error.message;
    }
}

// ============================================================================
// Settings Form (removed - now using individual save functions)
// ============================================================================

// ============================================================================
// Documents List
// ============================================================================

let debounceTimer = null;

function debounceLoadDocs() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(loadDocuments, 300);
}

function handleSearch(event) {
    if (event.key === 'Enter') {
        searchDocuments();
    }
}

async function loadDocuments() {
    const container = document.getElementById('documents-list');
    container.innerHTML = '<p class="placeholder-text">Loading documents...</p>';
    
    try {
        // Build query params
        const params = new URLSearchParams();
        
        const status = document.getElementById('filter-status').value;
        if (status) params.append('status', status);
        
        const sender = document.getElementById('filter-sender').value;
        if (sender) params.append('sender', sender);
        
        const dateFrom = document.getElementById('filter-date-from').value;
        if (dateFrom) params.append('date_from', dateFrom);
        
        const dateTo = document.getElementById('filter-date-to').value;
        if (dateTo) params.append('date_to', dateTo);
        
        const response = await apiGet(`/documents?${params.toString()}`);
        renderDocumentsList(response.documents);
    } catch (error) {
        console.error('Failed to load documents:', error);
        container.innerHTML = '<p class="placeholder-text error">Failed to load documents</p>';
    }
}

async function searchDocuments() {
    const query = document.getElementById('search-input').value.trim();
    if (!query) {
        loadDocuments();
        return;
    }
    
    const container = document.getElementById('documents-list');
    container.innerHTML = '<p class="placeholder-text">Searching...</p>';
    
    try {
        const response = await apiGet(`/documents/search?q=${encodeURIComponent(query)}`);
        renderDocumentsList(response.documents);
    } catch (error) {
        console.error('Failed to search documents:', error);
        container.innerHTML = '<p class="placeholder-text error">Search failed</p>';
    }
}

function clearFilters() {
    document.getElementById('filter-status').value = '';
    document.getElementById('filter-sender').value = '';
    document.getElementById('filter-date-from').value = '';
    document.getElementById('filter-date-to').value = '';
    document.getElementById('search-input').value = '';
    loadDocuments();
}

function renderDocumentsList(documents) {
    const container = document.getElementById('documents-list');
    
    if (!documents || documents.length === 0) {
        container.innerHTML = '<p class="placeholder-text">No documents found</p>';
        return;
    }
    
    const html = documents.map(doc => `
        <div class="document-card" onclick="openDocument(${doc.id})">
            <div class="document-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
            </div>
            <div class="document-info">
                <div class="document-title">${escapeHtml(doc.topic || 'Untitled Document')}</div>
                <div class="document-meta">
                    ${doc.sender ? `<span class="meta-item">${escapeHtml(doc.sender)}</span>` : ''}
                    ${doc.document_date ? `<span class="meta-item">${formatDate(doc.document_date)}</span>` : ''}
                    <span class="meta-item type">${escapeHtml(doc.document_type || 'document')}</span>
                </div>
            </div>
            <div class="document-status">
                <span class="status-badge status-${doc.status}">${formatStatus(doc.status)}</span>
            </div>
        </div>
    `).join('');
    
    container.innerHTML = html;
}

// ============================================================================
// Queue (Incoming Files)
// ============================================================================

async function loadQueue() {
    const container = document.getElementById('queue-list');
    container.innerHTML = '<p class="placeholder-text">Loading files...</p>';
    
    try {
        const response = await apiGet('/incoming/status');
        if (response.success) {
            renderIncomingFiles(response.files);
        } else {
            container.innerHTML = `<p class="placeholder-text error">${escapeHtml(response.message)}</p>`;
        }
    } catch (error) {
        console.error('Failed to load incoming files:', error);
        container.innerHTML = '<p class="placeholder-text error">Failed to load files</p>';
    }
}

function renderIncomingFiles(files) {
    const container = document.getElementById('queue-list');
    
    if (!files || files.length === 0) {
        container.innerHTML = '<p class="placeholder-text">No files in incoming folder</p>';
        return;
    }
    
    const html = files.map(file => {
        const status = getFileStatus(file);
        const statusClass = getStatusClass(status);
        const statusLabel = getStatusLabel(status);
        
        return `
            <div class="file-card">
                <div class="file-card-thumbnail" onclick="openPreview('${escapeAttr(file.path)}', '${escapeAttr(file.name)}')">
                    <img src="${escapeAttr(file.thumbnail_url)}" 
                         alt="${escapeAttr(file.name)}"
                         onerror="this.parentElement.innerHTML='<div class=\\'thumbnail-placeholder\\'><svg width=\\'48\\' height=\\'48\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'currentColor\\' stroke-width=\\'1\\'><path d=\\'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z\\'></path><polyline points=\\'14 2 14 8 20 8\\'></polyline></svg></div>'">
                    <div class="thumbnail-overlay">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                            <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                    </div>
                    <div class="file-card-status">
                        <span class="status-badge ${statusClass}">${statusLabel}</span>
                    </div>
                </div>
                <div class="file-card-body">
                    <div class="file-card-name" title="${escapeAttr(file.name)}">${escapeHtml(file.name)}</div>
                    <div class="file-card-meta">${formatFileSize(file.size)} • ${formatDateTime(file.modified)}</div>
                    <div class="file-card-info">
                        ${file.topic ? `<div class="info-topic">${escapeHtml(file.topic)}</div>` : ''}
                        ${file.sender ? `<div class="info-sender">${escapeHtml(file.sender)}</div>` : ''}
                        ${file.error_message ? `<div class="info-error" title="${escapeAttr(file.error_message)}">${escapeHtml(truncateText(file.error_message, 50))}</div>` : ''}
                        ${!file.topic && !file.sender && !file.error_message ? `<div class="info-pending">${getFileInfoText(file, status)}</div>` : ''}
                    </div>
                    <div class="file-card-actions">
                        ${renderFileActions(file, status)}
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = html;
}

function getFileStatus(file) {
    // Priority: document_status > queue_status
    if (file.document_status) {
        return file.document_status;
    }
    return file.queue_status || 'new';
}

function getStatusClass(status) {
    const classes = {
        'new': 'status-new',
        'queued': 'status-queued',
        'processing': 'status-processing',
        'completed': 'status-completed',
        'awaiting_approval': 'status-awaiting_approval',
        'approved': 'status-approved',
        'rejected': 'status-rejected',
        'error': 'status-error'
    };
    return classes[status] || 'status-pending';
}

function getStatusLabel(status) {
    const labels = {
        'new': 'New',
        'queued': 'Queued',
        'processing': 'Processing',
        'completed': 'Completed',
        'awaiting_approval': 'Awaiting Approval',
        'approved': 'Approved',
        'rejected': 'Rejected',
        'error': 'Error'
    };
    return labels[status] || status;
}

function getStatusDescription(status) {
    const descriptions = {
        'new': 'Not yet processed',
        'queued': 'Waiting in queue',
        'processing': 'Currently processing...',
        'completed': 'Processing complete',
        'awaiting_approval': 'Ready for review',
        'approved': 'Approved',
        'rejected': 'Rejected',
        'error': 'Processing failed'
    };
    return descriptions[status] || 'Pending processing';
}

function getFileInfoText(file, status) {
    // For files that have been processed but LLM returned no metadata
    if (status === 'awaiting_approval' || status === 'completed') {
        return 'Metadata extraction failed - click Reprocess';
    }
    // For other statuses, return status description
    return getStatusDescription(status);
}

function renderFileActions(file, status) {
    const path = escapeAttr(file.path);
    
    // Check if document was properly processed (has metadata from LLM)
    const hasMetadata = file.topic || file.sender;
    
    // Awaiting approval - show approve/reject only if properly processed
    if (status === 'awaiting_approval' && file.document_id && hasMetadata) {
        return `
            <button class="btn btn-success btn-small" onclick="approveDocument(${file.document_id}, event)">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                Approve
            </button>
            <button class="btn btn-danger btn-small" onclick="rejectDocument(${file.document_id}, event)">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
                Reject
            </button>
            <button class="btn btn-secondary btn-small" onclick="reprocessFile('${path}', event)">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="23 4 23 10 17 10"></polyline>
                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                </svg>
            </button>
        `;
    }
    
    // Awaiting approval but no metadata - offer reprocess
    if (status === 'awaiting_approval' && file.document_id && !hasMetadata) {
        return `
            <button class="btn btn-primary btn-small" onclick="reprocessFile('${path}', event)" style="flex: 1;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="23 4 23 10 17 10"></polyline>
                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                </svg>
                Reprocess
            </button>
        `;
    }
    
    // Error - show reprocess
    if (status === 'error') {
        return `
            <button class="btn btn-primary btn-small" onclick="reprocessFile('${path}', event)" style="flex: 1;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="23 4 23 10 17 10"></polyline>
                    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                </svg>
                Retry
            </button>
        `;
    }
    
    // Processing - show spinner/status
    if (status === 'processing') {
        return `
            <button class="btn btn-secondary btn-small" disabled style="flex: 1;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin">
                    <line x1="12" y1="2" x2="12" y2="6"></line>
                    <line x1="12" y1="18" x2="12" y2="22"></line>
                    <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
                    <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
                    <line x1="2" y1="12" x2="6" y2="12"></line>
                    <line x1="18" y1="12" x2="22" y2="12"></line>
                    <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line>
                    <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
                </svg>
                Processing...
            </button>
        `;
    }
    
    // Queued - show waiting
    if (status === 'queued') {
        return `
            <button class="btn btn-secondary btn-small" disabled style="flex: 1;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                Queued
            </button>
        `;
    }
    
    // New - show add to queue
    if (status === 'new') {
        return `
            <button class="btn btn-primary btn-small" onclick="queueFile('${path}', event)" style="flex: 1;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
                Queue
            </button>
        `;
    }
    
    // Completed/Approved/Rejected - show reprocess option
    return `
        <button class="btn btn-secondary btn-small" onclick="reprocessFile('${path}', event)" style="flex: 1;">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="23 4 23 10 17 10"></polyline>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
            </svg>
            Reprocess
        </button>
    `;
}

async function reprocessFile(filePath, event) {
    if (event) event.stopPropagation();
    
    try {
        const result = await apiPost(`/incoming/reprocess?path=${encodeURIComponent(filePath)}`);
        showNotification(result.message, 'success');
        loadQueue();
        loadStats();
    } catch (error) {
        console.error('Failed to reprocess file:', error);
        showNotification('Failed to reprocess file', 'error');
    }
}

async function queueFile(filePath, event) {
    if (event) event.stopPropagation();
    
    try {
        // Trigger a scan to add the file to queue
        await apiPost('/processing/scan');
        showNotification('File queued for processing', 'success');
        loadQueue();
        loadStats();
    } catch (error) {
        console.error('Failed to queue file:', error);
        showNotification('Failed to queue file', 'error');
    }
}

// ============================================================================
// Preview Modal
// ============================================================================

function openPreview(filePath, fileName) {
    const modal = document.getElementById('preview-modal');
    const title = document.getElementById('preview-title');
    const body = document.getElementById('preview-body');
    
    title.textContent = fileName;
    
    // Determine file type
    const ext = fileName.split('.').pop().toLowerCase();
    const imageExts = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff'];
    const pdfExts = ['pdf'];
    
    const previewUrl = `/api/incoming/preview?path=${encodeURIComponent(filePath)}`;
    
    if (imageExts.includes(ext)) {
        body.innerHTML = `<img src="${previewUrl}" alt="${escapeAttr(fileName)}">`;
    } else if (pdfExts.includes(ext)) {
        body.innerHTML = `<iframe src="${previewUrl}" title="${escapeAttr(fileName)}"></iframe>`;
    } else {
        body.innerHTML = `<div class="preview-error">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
            </svg>
            <p>Preview not available for this file type</p>
            <a href="${previewUrl}" download="${escapeAttr(fileName)}" class="btn btn-primary">Download File</a>
        </div>`;
    }
    
    modal.classList.add('active');
}

function closePreviewModal() {
    const modal = document.getElementById('preview-modal');
    const body = document.getElementById('preview-body');
    modal.classList.remove('active');
    // Clear content to stop any loading
    body.innerHTML = '';
}

// ============================================================================
// Utility Functions (Additional)
// ============================================================================

function escapeAttr(text) {
    if (!text) return '';
    return text.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (bytes >= 1024 && i < units.length - 1) {
        bytes /= 1024;
        i++;
    }
    return `${bytes.toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
}

function formatDateTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString('de-DE', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function truncateText(text, maxLength) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// ============================================================================
// Logs Page
// ============================================================================

let logsAutoRefreshInterval = null;
let currentLogData = [];

async function loadLogs() {
    const container = document.getElementById('logs-content');
    const countEl = document.getElementById('logs-count');
    const fileEl = document.getElementById('logs-file');
    
    const source = document.getElementById('log-source').value;
    const level = document.getElementById('log-level').value;
    
    container.innerHTML = '<p class="placeholder-text">Loading logs...</p>';
    
    try {
        const response = await apiGet(`/logs/${source}?lines=1000&level=${level}`);
        
        if (!response.exists) {
            container.innerHTML = `<p class="placeholder-text">${escapeHtml(response.message)}</p>`;
            countEl.textContent = '0 entries';
            fileEl.textContent = response.file || '';
            currentLogData = [];
            return;
        }
        
        currentLogData = response.lines || [];
        countEl.textContent = `${currentLogData.length} entries`;
        fileEl.textContent = response.file || '';
        
        renderLogs(currentLogData);
        
    } catch (error) {
        console.error('Failed to load logs:', error);
        container.innerHTML = '<p class="placeholder-text error">Failed to load logs</p>';
        countEl.textContent = '0 entries';
    }
}

function renderLogs(logEntries) {
    const container = document.getElementById('logs-content');
    
    if (!logEntries || logEntries.length === 0) {
        container.innerHTML = '<p class="placeholder-text">No log entries found</p>';
        return;
    }
    
    const html = logEntries.map(entry => {
        const levelClass = `log-${entry.level}`;
        return `<div class="log-entry ${levelClass}">${escapeHtml(entry.text)}</div>`;
    }).join('');
    
    container.innerHTML = html;
    
    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

function filterLogs() {
    // Reload logs with new filter
    loadLogs();
}

function toggleLogAutoRefresh() {
    const checkbox = document.getElementById('log-auto-refresh');
    
    if (checkbox.checked) {
        // Start auto-refresh every 3 seconds
        logsAutoRefreshInterval = setInterval(loadLogs, 3000);
    } else {
        // Stop auto-refresh
        if (logsAutoRefreshInterval) {
            clearInterval(logsAutoRefreshInterval);
            logsAutoRefreshInterval = null;
        }
    }
}

// ============================================================================
// Document Actions
// ============================================================================

async function openDocument(docId) {
    try {
        const doc = await apiGet(`/documents/${docId}`);
        showDocumentModal(doc);
    } catch (error) {
        console.error('Failed to load document:', error);
        showNotification('Failed to load document details', 'error');
    }
}

function showDocumentModal(doc) {
    const modal = document.getElementById('document-modal');
    const title = document.getElementById('modal-title');
    const body = document.getElementById('modal-body');
    const footer = document.getElementById('modal-footer');
    
    title.textContent = doc.topic || 'Document Details';
    
    body.innerHTML = `
        <div class="document-detail">
            <div class="detail-row">
                <label>Status:</label>
                <span class="status-badge status-${doc.status}">${formatStatus(doc.status)}</span>
            </div>
            <div class="detail-row">
                <label>Sender:</label>
                <span>${escapeHtml(doc.sender || '-')}</span>
            </div>
            <div class="detail-row">
                <label>Receiver:</label>
                <span>${escapeHtml(doc.receiver || '-')}</span>
            </div>
            <div class="detail-row">
                <label>Date:</label>
                <span>${doc.document_date ? formatDate(doc.document_date) : '-'}</span>
            </div>
            <div class="detail-row">
                <label>Type:</label>
                <span>${escapeHtml(doc.document_type || '-')}</span>
            </div>
            <div class="detail-row">
                <label>Language:</label>
                <span>${doc.language || '-'}</span>
            </div>
            ${doc.due_date ? `
                <div class="detail-row">
                    <label>Due Date:</label>
                    <span>${formatDate(doc.due_date)}</span>
                </div>
            ` : ''}
            <div class="detail-row">
                <label>Summary:</label>
                <span>${escapeHtml(doc.summary || '-')}</span>
            </div>
            ${doc.iban || doc.bic ? `
                <div class="detail-section">
                    <h4>Bank Details</h4>
                    ${doc.iban ? `
                        <div class="detail-row">
                            <label>IBAN:</label>
                            <span class="bank-detail">${escapeHtml(doc.iban)}</span>
                        </div>
                    ` : ''}
                    ${doc.bic ? `
                        <div class="detail-row">
                            <label>BIC:</label>
                            <span class="bank-detail">${escapeHtml(doc.bic)}</span>
                        </div>
                    ` : ''}
                </div>
            ` : ''}
            ${doc.identifiers && doc.identifiers.length > 0 ? `
                <div class="detail-section">
                    <h4>Reference Numbers / Identifiers</h4>
                    <div class="identifiers-list">
                        ${doc.identifiers.map(ident => `
                            <div class="identifier-badge" title="Click to search for related documents" onclick="searchByIdentifier('${escapeAttr(ident.identifier_value)}', '${escapeAttr(ident.identifier_type)}')">
                                <span class="ident-type">${escapeHtml(ident.identifier_type)}</span>
                                <span class="ident-value">${escapeHtml(ident.identifier_value)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
            <div class="detail-section">
                <h4>Related Documents</h4>
                <div id="related-docs-list" class="related-docs-list">
                    <p class="placeholder-text loading">Loading related documents...</p>
                </div>
            </div>
            ${doc.processing_error ? `
                <div class="detail-row error">
                    <label>Error:</label>
                    <span>${escapeHtml(doc.processing_error)}</span>
                </div>
            ` : ''}
            <div class="detail-section">
                <h4>Pages (${doc.pages ? doc.pages.length : 0})</h4>
                <div class="pages-list">
                    ${doc.pages && doc.pages.length > 0 ? doc.pages.map(page => `
                        <div class="page-item">
                            <div class="page-header">
                                <span>Page ${page.page_number}</span>
                                <span class="confidence">Confidence: ${(page.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div class="page-text">${escapeHtml(page.ocr_text || 'No text extracted').substring(0, 500)}${page.ocr_text && page.ocr_text.length > 500 ? '...' : ''}</div>
                        </div>
                    `).join('') : '<p>No pages</p>'}
                </div>
            </div>
        </div>
    `;
    
    // Footer actions based on status
    let actions = '';
    if (doc.status === 'awaiting_approval') {
        actions = `
            <button class="btn btn-success" onclick="approveDocument(${doc.id}); closeModal();">Approve</button>
            <button class="btn btn-danger" onclick="rejectDocument(${doc.id}); closeModal();">Reject</button>
        `;
    } else if (doc.status === 'error') {
        actions = `
            <button class="btn btn-primary" onclick="reprocessDocument(${doc.id}); closeModal();">Reprocess</button>
        `;
    }
    actions += `<button class="btn btn-secondary" onclick="closeModal()">Close</button>`;
    footer.innerHTML = actions;
    
    modal.classList.add('active');
    
    // Load related documents asynchronously
    loadRelatedDocuments(doc.id);
}

function closeModal() {
    document.getElementById('document-modal').classList.remove('active');
}

// ============================================================================
// Related Documents and Identifier Search
// ============================================================================

async function loadRelatedDocuments(docId) {
    const container = document.getElementById('related-docs-list');
    if (!container) return;
    
    try {
        const data = await apiGet(`/documents/${docId}/related`);
        
        if (data.related_count === 0) {
            container.innerHTML = '<p class="placeholder-text">No related documents found</p>';
            return;
        }
        
        const html = data.related_documents.map(doc => `
            <div class="related-doc-item" onclick="openDocument(${doc.id}); event.stopPropagation();">
                <div class="related-doc-main">
                    <span class="related-doc-title">${escapeHtml(doc.topic || 'Untitled')}</span>
                    <span class="related-doc-meta">
                        ${doc.sender ? escapeHtml(doc.sender) : ''}
                        ${doc.sender && doc.document_date ? ' - ' : ''}
                        ${doc.document_date ? formatDate(doc.document_date) : ''}
                    </span>
                </div>
                <div class="related-doc-match">
                    <span class="match-label">Matching:</span>
                    ${doc.matching_identifiers.map(ident => `
                        <span class="match-badge">${escapeHtml(ident.type)}: ${escapeHtml(ident.value)}</span>
                    `).join('')}
                </div>
            </div>
        `).join('');
        
        container.innerHTML = `
            <p class="related-count">${data.related_count} related document${data.related_count !== 1 ? 's' : ''} found</p>
            ${html}
        `;
        
    } catch (error) {
        console.error('Failed to load related documents:', error);
        container.innerHTML = '<p class="placeholder-text">Failed to load related documents</p>';
    }
}

async function searchByIdentifier(value, type) {
    // Close the current modal
    closeModal();
    
    // Navigate to documents page
    navigateTo('documents');
    
    // Update search input and trigger search
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.value = value;
    }
    
    // Perform the search using the identifier search endpoint
    const container = document.getElementById('documents-list');
    container.innerHTML = '<p class="placeholder-text">Searching...</p>';
    
    try {
        const params = new URLSearchParams();
        params.append('value', value);
        if (type) {
            params.append('identifier_type', type);
        }
        
        const response = await apiGet(`/identifiers/search?${params.toString()}`);
        
        if (response.documents && response.documents.length > 0) {
            renderDocumentsList(response.documents);
            showNotification(`Found ${response.documents.length} document(s) with identifier "${value}"`, 'success');
        } else {
            container.innerHTML = `<p class="placeholder-text">No documents found with identifier "${escapeHtml(value)}"</p>`;
        }
    } catch (error) {
        console.error('Failed to search by identifier:', error);
        container.innerHTML = '<p class="placeholder-text error">Search failed</p>';
        showNotification('Failed to search by identifier', 'error');
    }
}

async function approveDocument(docId, event) {
    if (event) event.stopPropagation();
    
    try {
        await apiPost(`/documents/${docId}/approve`);
        showNotification('Document approved', 'success');
        loadStats();
        loadQueue();
        loadDocuments();
    } catch (error) {
        console.error('Failed to approve document:', error);
        showNotification('Failed to approve document', 'error');
    }
}

async function rejectDocument(docId, event) {
    if (event) event.stopPropagation();
    
    // Open modal instead of using prompt()
    openRejectionModal(docId);
}

async function reprocessDocument(docId) {
    try {
        await apiPost(`/documents/${docId}/reprocess`);
        showNotification('Document queued for reprocessing', 'info');
        loadStats();
        loadDocuments();
    } catch (error) {
        console.error('Failed to reprocess document:', error);
        showNotification('Failed to reprocess document', 'error');
    }
}

// ============================================================================
// Processing Actions
// ============================================================================

async function triggerScan() {
    try {
        const result = await apiPost('/processing/scan');
        loadStats();
        // Use appropriate notification type based on result
        if (result.newly_added > 0) {
            showNotification(result.message, 'success');
        } else if (result.total_pending > 0) {
            showNotification(result.message, 'info');
        } else if (result.files && result.files.length > 0) {
            // Files found but in error state, inform user
            showNotification(`${result.files.length} file(s) need attention (errors)`, 'warning');
        } else {
            showNotification(result.message, 'info');
        }
    } catch (error) {
        console.error('Failed to trigger scan:', error);
        showNotification('Failed to trigger scan', 'error');
    }
}

async function processQueue() {
    try {
        const result = await apiPost('/processing/process-queue');
        loadStats();
        loadQueue();
        // Use appropriate notification type based on result
        if (result.processed_count > 0) {
            showNotification(result.message, 'success');
        } else {
            showNotification(result.message, 'info');
        }
    } catch (error) {
        console.error('Failed to process queue:', error);
        showNotification('Failed to process queue', 'error');
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateStr) {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString('de-DE', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
    });
}

function formatStatus(status) {
    const statusMap = {
        'pending': 'Pending',
        'processing': 'Processing',
        'awaiting_approval': 'Awaiting Approval',
        'approved': 'Approved',
        'rejected': 'Rejected',
        'error': 'Error'
    };
    return statusMap[status] || status;
}

// ============================================================================
// Expose functions to global scope
// ============================================================================

window.navigateTo = navigateTo;
window.testConnection = testConnection;
window.loadDocuments = loadDocuments;
window.searchDocuments = searchDocuments;
window.clearFilters = clearFilters;
window.handleSearch = handleSearch;
window.debounceLoadDocs = debounceLoadDocs;
window.openDocument = openDocument;
window.closeModal = closeModal;
window.approveDocument = approveDocument;
window.rejectDocument = rejectDocument;
window.reprocessDocument = reprocessDocument;
window.triggerScan = triggerScan;
window.processQueue = processQueue;
window.toggleProcessingMode = toggleProcessingMode;
window.cancelProcessing = cancelProcessing;
window.toggleShareEdit = toggleShareEdit;
window.cancelShareEdit = cancelShareEdit;
window.saveShareConfig = saveShareConfig;
window.saveSchedulerConfig = saveSchedulerConfig;
window.saveProcessingMode = saveProcessingMode;
window.saveLLMConfig = saveLLMConfig;
window.resetLLMConfig = resetLLMConfig;
window.saveProcessingConfig = saveProcessingConfig;
window.closeRejectionModal = closeRejectionModal;
window.confirmRejection = confirmRejection;
window.openPreview = openPreview;
window.closePreviewModal = closePreviewModal;
window.reprocessFile = reprocessFile;
window.queueFile = queueFile;
window.loadLogs = loadLogs;
window.filterLogs = filterLogs;
window.toggleLogAutoRefresh = toggleLogAutoRefresh;
// Model management functions
window.loadModels = loadModels;
window.switchModel = switchModel;
window.downloadModel = downloadModel;
window.deleteModel = deleteModel;
// Related documents and identifier search
window.loadRelatedDocuments = loadRelatedDocuments;
window.searchByIdentifier = searchByIdentifier;
// Vision LLM configuration
window.saveVisionLLMConfig = saveVisionLLMConfig;
window.resetVisionLLMPrompts = resetVisionLLMPrompts;
window.copyToClipboard = copyToClipboard;

// ============================================================================
// Initialize
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
    
    // Start SSE connection for real-time updates
    SSE.connect();
    
    // Start progress bar polling (as backup/complement to SSE)
    startProgressPolling();
    
    // Vision LLM max pixels slider live update
    const maxPixelsSlider = document.getElementById('vision-llm-max-pixels');
    if (maxPixelsSlider) {
        maxPixelsSlider.addEventListener('input', (e) => {
            updateMaxPixelsDisplay(e.target.value);
        });
    }
    
    // Close modals on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
            closePreviewModal();
            closeRejectionModal();
        }
    });
    
    // Close modal on backdrop click
    document.getElementById('document-modal').addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) closeModal();
    });
    
    // Close preview modal on backdrop click
    document.getElementById('preview-modal').addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) closePreviewModal();
    });
    
    // Close rejection modal on backdrop click
    document.getElementById('rejection-modal').addEventListener('click', (e) => {
        if (e.target.classList.contains('modal')) closeRejectionModal();
    });
});
