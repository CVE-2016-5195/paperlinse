// Paperlinse Frontend Application

// Navigation
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
}

// API calls
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

// Load configuration
async function loadConfig() {
    try {
        const config = await apiGet('/config');
        
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
        
        // Update dashboard based on config
        updateDashboard(config);
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

function updateDashboard(config) {
    const setupPrompt = document.getElementById('setup-prompt');
    const hasConfig = config.incoming_share.path && config.storage.path;
    
    if (hasConfig) {
        setupPrompt.style.display = 'none';
    } else {
        setupPrompt.style.display = 'block';
    }
    
    // Placeholder counts - will be implemented later
    document.getElementById('pending-count').textContent = '0';
    document.getElementById('processed-count').textContent = '0';
    document.getElementById('total-count').textContent = '0';
}

// Test connection
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

// Save configuration
document.getElementById('settings-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const statusEl = document.getElementById('save-status');
    statusEl.className = 'save-status';
    statusEl.textContent = 'Saving...';
    
    const config = {
        incoming_share: {
            path: document.getElementById('incoming-path').value,
            poll_interval_seconds: parseInt(document.getElementById('poll-interval').value) || 30,
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
        statusEl.textContent = 'Configuration saved successfully!';
        setTimeout(() => { statusEl.textContent = ''; }, 3000);
        
        // Reload config to update dashboard
        loadConfig();
    } catch (error) {
        statusEl.className = 'save-status error';
        statusEl.textContent = 'Failed to save: ' + error.message;
    }
});

// Expose functions to global scope for onclick handlers
window.navigateTo = navigateTo;
window.testConnection = testConnection;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
});
