/**
 * API Client Module
 * 
 * Centralized API communication layer with consistent error handling.
 * All API calls should go through this module.
 */

const API_BASE = '/api';

/**
 * Custom error class for API errors with additional context
 */
export class ApiError extends Error {
    constructor(message, status, data = null) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.data = data;
    }
}

/**
 * Base fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
        });
        
        if (!response.ok) {
            let errorData = null;
            try {
                errorData = await response.json();
            } catch {
                // Response wasn't JSON
            }
            throw new ApiError(
                errorData?.detail || `API error: ${response.status}`,
                response.status,
                errorData
            );
        }
        
        // Handle empty responses (204 No Content)
        if (response.status === 204) {
            return null;
        }
        
        return response.json();
    } catch (error) {
        if (error instanceof ApiError) {
            throw error;
        }
        // Network error or other fetch failure
        throw new ApiError(
            error.message || 'Network error',
            0,
            null
        );
    }
}

// =============================================================================
// HTTP Methods
// =============================================================================

export async function get(endpoint) {
    return apiFetch(endpoint);
}

export async function post(endpoint, data = null) {
    return apiFetch(endpoint, {
        method: 'POST',
        body: data ? JSON.stringify(data) : undefined,
    });
}

export async function put(endpoint, data = null) {
    return apiFetch(endpoint, {
        method: 'PUT',
        body: data ? JSON.stringify(data) : undefined,
    });
}

export async function del(endpoint) {
    return apiFetch(endpoint, {
        method: 'DELETE',
    });
}

// =============================================================================
// Config API
// =============================================================================

export const config = {
    async get() {
        return get('/config');
    },
    
    async save(data) {
        return post('/config', data);
    },
    
    async getLLM() {
        return get('/config/llm');
    },
    
    async saveLLM(data) {
        return post('/config/llm', data);
    },
    
    async resetLLM() {
        return post('/config/llm/reset');
    },
    
    async getProcessing() {
        return get('/config/processing');
    },
    
    async saveProcessing(data) {
        return post('/config/processing', data);
    },
};

// =============================================================================
// Documents API
// =============================================================================

export const documents = {
    async list(params = {}) {
        const query = new URLSearchParams();
        if (params.status) query.append('status', params.status);
        if (params.sender) query.append('sender', params.sender);
        if (params.search) query.append('search', params.search);
        if (params.from_date) query.append('from_date', params.from_date);
        if (params.to_date) query.append('to_date', params.to_date);
        if (params.limit) query.append('limit', params.limit);
        if (params.offset) query.append('offset', params.offset);
        
        const queryString = query.toString();
        return get(`/documents${queryString ? `?${queryString}` : ''}`);
    },
    
    async get(id) {
        return get(`/documents/${id}`);
    },
    
    async approve(id) {
        return post(`/documents/${id}/approve`);
    },
    
    async reject(id, reason = '') {
        return post(`/documents/${id}/reject?reason=${encodeURIComponent(reason)}`);
    },
    
    async reprocess(id) {
        return post(`/documents/${id}/reprocess`);
    },
    
    async delete(id) {
        return del(`/documents/${id}`);
    },
};

// =============================================================================
// Queue API
// =============================================================================

export const queue = {
    async getStatus() {
        return get('/incoming/status');
    },
    
    async reprocess(filePath) {
        return post('/incoming/reprocess', { file_path: filePath });
    },
};

// =============================================================================
// Processing API
// =============================================================================

export const processing = {
    async getProgress() {
        return get('/processing/progress');
    },
    
    async getMode() {
        return get('/processing/mode');
    },
    
    async setMode(autoApprove) {
        return put('/processing/mode', { auto_approve: autoApprove });
    },
    
    async scan() {
        return post('/processing/scan');
    },
    
    async processQueue() {
        return post('/processing/process-queue');
    },
    
    async processFile(filePath) {
        return post('/processing/process', { file_path: filePath });
    },
};

// =============================================================================
// Stats API
// =============================================================================

export const stats = {
    async get() {
        return get('/stats');
    },
};

// =============================================================================
// Models API
// =============================================================================

export const models = {
    async list() {
        return get('/models');
    },
    
    async switch(modelId, downloadIfMissing = false) {
        return post(`/models/${encodeURIComponent(modelId)}/switch`, {
            download_if_missing: downloadIfMissing,
        });
    },
    
    async delete(modelId) {
        return del(`/models/${encodeURIComponent(modelId)}`);
    },
    
    async getPullStatus(modelId) {
        return get(`/models/${encodeURIComponent(modelId)}/pull/status`);
    },
    
    /**
     * Pull a model with progress streaming
     * @param {string} modelId - Model ID to pull
     * @param {function} onProgress - Callback for progress updates
     * @returns {Promise} Resolves when pull completes
     */
    async pull(modelId, onProgress) {
        const response = await fetch(`${API_BASE}/models/${encodeURIComponent(modelId)}/pull`, {
            method: 'POST',
        });
        
        if (!response.ok) {
            throw new ApiError(`Failed to start model pull: ${response.status}`, response.status);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Process complete SSE messages
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (onProgress) {
                            onProgress(data);
                        }
                        if (data.status === 'success' || data.status === 'error') {
                            return data;
                        }
                    } catch (e) {
                        console.warn('Failed to parse SSE data:', line);
                    }
                }
            }
        }
    },
};

// =============================================================================
// Logs API
// =============================================================================

export const logs = {
    async get(source = 'paperlinse', lines = 200) {
        return get(`/logs/${source}?lines=${lines}`);
    },
};

// =============================================================================
// Test Connection API
// =============================================================================

export const connection = {
    async test(path, credentials) {
        return post('/test-connection', { path, credentials });
    },
};

// =============================================================================
// Thumbnails & Previews
// =============================================================================

export function getThumbnailUrl(filePath) {
    return `${API_BASE}/incoming/thumbnail?file_path=${encodeURIComponent(filePath)}`;
}

export function getPreviewUrl(filePath, page = 1) {
    return `${API_BASE}/incoming/preview?file_path=${encodeURIComponent(filePath)}&page=${page}`;
}

export function getDocumentThumbnailUrl(docId) {
    return `${API_BASE}/documents/${docId}/thumbnail`;
}
