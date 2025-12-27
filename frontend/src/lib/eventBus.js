/**
 * EventBus Module
 * 
 * Central event system for decoupled component communication.
 * Components can subscribe to events and emit events without direct references.
 */

class EventBus {
    constructor() {
        this.listeners = new Map();
    }
    
    /**
     * Subscribe to an event
     * @param {string} event - Event name
     * @param {function} callback - Handler function
     * @returns {function} Unsubscribe function
     */
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);
        
        // Return unsubscribe function
        return () => this.off(event, callback);
    }
    
    /**
     * Subscribe to an event (one-time)
     * @param {string} event - Event name
     * @param {function} callback - Handler function
     */
    once(event, callback) {
        const wrapper = (...args) => {
            this.off(event, wrapper);
            callback(...args);
        };
        this.on(event, wrapper);
    }
    
    /**
     * Unsubscribe from an event
     * @param {string} event - Event name
     * @param {function} callback - Handler function to remove
     */
    off(event, callback) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).delete(callback);
        }
    }
    
    /**
     * Emit an event
     * @param {string} event - Event name
     * @param {*} data - Event data
     */
    emit(event, data = null) {
        if (this.listeners.has(event)) {
            for (const callback of this.listeners.get(event)) {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`EventBus: Error in listener for "${event}":`, error);
                }
            }
        }
    }
    
    /**
     * Remove all listeners for an event (or all events)
     * @param {string} [event] - Event name (optional, removes all if not specified)
     */
    clear(event = null) {
        if (event) {
            this.listeners.delete(event);
        } else {
            this.listeners.clear();
        }
    }
}

// Global event bus instance
export const eventBus = new EventBus();

// =============================================================================
// Event Names (constants to avoid typos)
// =============================================================================

export const Events = {
    // Navigation
    PAGE_CHANGED: 'page:changed',
    
    // Data refresh events (triggered when data needs to be reloaded)
    REFRESH_STATS: 'refresh:stats',
    REFRESH_QUEUE: 'refresh:queue',
    REFRESH_DOCUMENTS: 'refresh:documents',
    REFRESH_ALL: 'refresh:all',
    
    // Processing events (from SSE)
    PROCESSING_STARTED: 'processing:started',
    PROCESSING_PROGRESS: 'processing:progress',
    PROCESSING_COMPLETED: 'processing:completed',
    PROCESSING_ERROR: 'processing:error',
    
    // Queue events (from SSE)
    QUEUE_UPDATED: 'queue:updated',
    
    // Document events (from SSE)
    DOCUMENT_CREATED: 'document:created',
    DOCUMENT_UPDATED: 'document:updated',
    DOCUMENT_APPROVED: 'document:approved',
    DOCUMENT_REJECTED: 'document:rejected',
    
    // Stats events (from SSE)
    STATS_UPDATED: 'stats:updated',
    
    // Scan events
    SCAN_STARTED: 'scan:started',
    SCAN_COMPLETED: 'scan:completed',
    
    // Connection events
    SSE_CONNECTED: 'sse:connected',
    SSE_DISCONNECTED: 'sse:disconnected',
    SSE_ERROR: 'sse:error',
    
    // UI events
    NOTIFICATION: 'ui:notification',
    MODAL_OPEN: 'ui:modal:open',
    MODAL_CLOSE: 'ui:modal:close',
};

export default eventBus;
