/**
 * SSE (Server-Sent Events) Client Module
 * 
 * Manages the connection to the backend SSE endpoint for real-time updates.
 * Automatically reconnects on disconnection with exponential backoff.
 */

import { eventBus, Events } from './eventBus.js';

class SSEClient {
    constructor() {
        this.eventSource = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.baseReconnectDelay = 1000; // 1 second
        this.maxReconnectDelay = 30000; // 30 seconds
        this.reconnectTimer = null;
        this.isConnecting = false;
        this.isConnected = false;
    }
    
    /**
     * Connect to the SSE endpoint
     */
    connect() {
        if (this.isConnecting || this.isConnected) {
            return;
        }
        
        this.isConnecting = true;
        console.log('SSE: Connecting to /api/events...');
        
        try {
            this.eventSource = new EventSource('/api/events');
            
            this.eventSource.onopen = () => {
                console.log('SSE: Connected');
                this.isConnecting = false;
                this.isConnected = true;
                this.reconnectAttempts = 0;
                eventBus.emit(Events.SSE_CONNECTED);
            };
            
            this.eventSource.onmessage = (event) => {
                this.handleMessage(event);
            };
            
            this.eventSource.onerror = (error) => {
                console.error('SSE: Connection error', error);
                this.isConnecting = false;
                this.isConnected = false;
                
                // Close the current connection
                if (this.eventSource) {
                    this.eventSource.close();
                    this.eventSource = null;
                }
                
                eventBus.emit(Events.SSE_DISCONNECTED);
                
                // Schedule reconnection
                this.scheduleReconnect();
            };
            
        } catch (error) {
            console.error('SSE: Failed to create EventSource', error);
            this.isConnecting = false;
            this.scheduleReconnect();
        }
    }
    
    /**
     * Disconnect from the SSE endpoint
     */
    disconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        
        this.isConnecting = false;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        
        console.log('SSE: Disconnected');
    }
    
    /**
     * Schedule a reconnection attempt with exponential backoff
     */
    scheduleReconnect() {
        if (this.reconnectTimer) {
            return; // Already scheduled
        }
        
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('SSE: Max reconnection attempts reached');
            eventBus.emit(Events.SSE_ERROR, { 
                message: 'Unable to connect to server. Please refresh the page.' 
            });
            return;
        }
        
        // Exponential backoff with jitter
        const delay = Math.min(
            this.baseReconnectDelay * Math.pow(2, this.reconnectAttempts) + Math.random() * 1000,
            this.maxReconnectDelay
        );
        
        this.reconnectAttempts++;
        console.log(`SSE: Reconnecting in ${Math.round(delay / 1000)}s (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, delay);
    }
    
    /**
     * Handle incoming SSE message
     */
    handleMessage(event) {
        try {
            const payload = JSON.parse(event.data);
            const { type, data, timestamp } = payload;
            
            // Map SSE event types to EventBus events
            switch (type) {
                case 'processing_started':
                    eventBus.emit(Events.PROCESSING_STARTED, data);
                    break;
                    
                case 'processing_progress':
                    eventBus.emit(Events.PROCESSING_PROGRESS, data);
                    break;
                    
                case 'processing_completed':
                    eventBus.emit(Events.PROCESSING_COMPLETED, data);
                    // Also trigger refreshes
                    eventBus.emit(Events.REFRESH_QUEUE);
                    eventBus.emit(Events.REFRESH_STATS);
                    eventBus.emit(Events.REFRESH_DOCUMENTS);
                    break;
                    
                case 'processing_error':
                    eventBus.emit(Events.PROCESSING_ERROR, data);
                    eventBus.emit(Events.REFRESH_QUEUE);
                    eventBus.emit(Events.REFRESH_STATS);
                    break;
                    
                case 'queue_updated':
                    eventBus.emit(Events.QUEUE_UPDATED, data);
                    eventBus.emit(Events.REFRESH_QUEUE);
                    break;
                    
                case 'document_created':
                    eventBus.emit(Events.DOCUMENT_CREATED, data);
                    eventBus.emit(Events.REFRESH_DOCUMENTS);
                    break;
                    
                case 'document_updated':
                    eventBus.emit(Events.DOCUMENT_UPDATED, data);
                    eventBus.emit(Events.REFRESH_DOCUMENTS);
                    break;
                    
                case 'document_approved':
                    eventBus.emit(Events.DOCUMENT_APPROVED, data);
                    eventBus.emit(Events.REFRESH_DOCUMENTS);
                    eventBus.emit(Events.REFRESH_QUEUE);
                    eventBus.emit(Events.REFRESH_STATS);
                    break;
                    
                case 'document_rejected':
                    eventBus.emit(Events.DOCUMENT_REJECTED, data);
                    eventBus.emit(Events.REFRESH_DOCUMENTS);
                    eventBus.emit(Events.REFRESH_QUEUE);
                    eventBus.emit(Events.REFRESH_STATS);
                    break;
                    
                case 'stats_updated':
                    eventBus.emit(Events.STATS_UPDATED, data);
                    eventBus.emit(Events.REFRESH_STATS);
                    break;
                    
                case 'scan_started':
                    eventBus.emit(Events.SCAN_STARTED, data);
                    break;
                    
                case 'scan_completed':
                    eventBus.emit(Events.SCAN_COMPLETED, data);
                    eventBus.emit(Events.REFRESH_QUEUE);
                    eventBus.emit(Events.REFRESH_STATS);
                    break;
                    
                case 'heartbeat':
                    // Heartbeat - connection is alive, no action needed
                    break;
                    
                default:
                    console.warn('SSE: Unknown event type:', type);
            }
            
        } catch (error) {
            console.error('SSE: Failed to parse message:', event.data, error);
        }
    }
    
    /**
     * Get connection status
     */
    get status() {
        if (this.isConnected) return 'connected';
        if (this.isConnecting) return 'connecting';
        return 'disconnected';
    }
}

// Global SSE client instance
export const sseClient = new SSEClient();

export default sseClient;
