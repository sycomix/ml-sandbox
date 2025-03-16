/**
 * ML Sandbox - Main JavaScript
 * Handles core functionality of the ML Sandbox application
 */

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Socket.IO connection
    const socket = io();
    
    // Initialize jsPlumb instance
    const jsPlumbInstance = jsPlumb.getInstance({
        Endpoint: ["Dot", { radius: 4 }],
        Connector: ["Bezier", { curviness: 50 }],
        PaintStyle: { stroke: "#4a6bff", strokeWidth: 2 },
        HoverPaintStyle: { stroke: "#3a5bef", strokeWidth: 3 },
        ConnectionOverlays: [
            ["Arrow", { location: 1, width: 10, length: 10, id: "arrow" }]
        ],
        Container: "canvas"
    });
    
    // Global variables
    let selectedComponent = null;
    let componentCounter = 0;
    let workflow = {
        nodes: [],
        connections: []
    };
    let isExecuting = false;
    let settings = loadSettings();
    
    // Initialize the application
    initializeApp();
    
    /**
     * Initialize the application
     */
    function initializeApp() {
        // Load components from the server
        loadComponents();
        
        // Initialize drag and drop functionality
        initializeDragAndDrop();
        
        // Initialize event listeners
        initializeEventListeners();
        
        // Initialize socket events
        initializeSocketEvents();
        
        // Apply user settings
        applySettings();
    }
    
    /**
     * Load components from the server
     */
    function loadComponents() {
        fetch('/api/components')
            .then(response => response.json())
            .then(data => {
                // Store components data globally
                window.mlComponents = data;
                
                // Display the first category (layers) by default
                displayComponentsByCategory('layers');
            })
            .catch(error => {
                showError('Failed to load components: ' + error.message);
            });
    }
    
    /**
     * Display components by category
     * @param {string} category - The category to display
     */
    function displayComponentsByCategory(category) {
        const componentsContainer = document.getElementById('components-container');
        componentsContainer.innerHTML = '';
        
        if (!window.mlComponents || !window.mlComponents[category]) {
            componentsContainer.innerHTML = '<p>No components found in this category.</p>';
            return;
        }
        
        window.mlComponents[category].forEach(component => {
            const componentElement = document.createElement('div');
            componentElement.className = 'component-item';
            componentElement.setAttribute('data-component-id', component.id);
            componentElement.setAttribute('data-component-category', category);
            componentElement.setAttribute('draggable', 'true');
            
            componentElement.innerHTML = `
                <h4>${component.name}</h4>
                <p>${component.category}</p>
            `;
            
            componentsContainer.appendChild(componentElement);
        });
        
        // Update active category
        document.querySelectorAll('.category').forEach(el => {
            el.classList.remove('active');
            if (el.getAttribute('data-category') === category) {
                el.classList.add('active');
            }
        });
    }
    
    /**
     * Initialize drag and drop functionality
     */
    function initializeDragAndDrop() {
        // Make components draggable
        document.addEventListener('dragstart', function(event) {
            if (event.target.classList.contains('component-item')) {
                event.dataTransfer.setData('component-id', event.target.getAttribute('data-component-id'));
                event.dataTransfer.setData('component-category', event.target.getAttribute('data-component-category'));
            }
        });
        
        // Allow dropping on the canvas
        const canvas = document.getElementById('canvas');
        
        canvas.addEventListener('dragover', function(event) {
            event.preventDefault();
        });
        
        canvas.addEventListener('drop', function(event) {
            event.preventDefault();
            
            const componentId = event.dataTransfer.getData('component-id');
            const componentCategory = event.dataTransfer.getData('component-category');
            
            if (componentId && componentCategory) {
                // Get component data
                const componentData = window.mlComponents[componentCategory].find(c => c.id === componentId);
                
                if (componentData) {
                    // Create a new component on the canvas
                    createCanvasComponent(componentData, event.clientX, event.clientY);
                }
            }
        });
    }
    
    /**
     * Create a new component on the canvas
     * @param {Object} componentData - The component data
     * @param {number} clientX - The X position of the drop
     * @param {number} clientY - The Y position of the drop
     */
    function createCanvasComponent(componentData, clientX, clientY) {
        // Get canvas position and scroll
        const canvas = document.getElementById('canvas');
        const canvasRect = canvas.getBoundingClientRect();
        const scrollLeft = canvas.scrollLeft;
        const scrollTop = canvas.scrollTop;
        
        // Calculate position relative to the canvas
        const x = clientX - canvasRect.left + scrollLeft;
        const y = clientY - canvasRect.top + scrollTop;
        
        // Create a unique ID for this component instance
        const instanceId = `${componentData.id}-${componentCounter++}`;
        
        // Create the component element
        const componentElement = document.createElement('div');
        componentElement.className = 'canvas-component';
        componentElement.id = instanceId;
        componentElement.setAttribute('data-component-id', componentData.id);
        componentElement.setAttribute('data-component-category', componentData.category);
        componentElement.style.left = `${x}px`;
        componentElement.style.top = `${y}px`;
        
        // Create component HTML
        componentElement.innerHTML = `
            <div class="header">
                <h4>${componentData.name}</h4>
                <div class="controls">
                    <button class="config-btn" title="Configure"><i class="fas fa-cog"></i></button>
                    <button class="delete-btn" title="Delete"><i class="fas fa-trash"></i></button>
                </div>
            </div>
            <div class="content">
                ${componentData.category}
            </div>
            <div class="ports">
                <div class="input-ports">
                    <div class="port input" data-port-type="input" data-port-name="input"></div>
                </div>
                <div class="output-ports">
                    <div class="port output" data-port-type="output" data-port-name="output"></div>
                </div>
            </div>
        `;
        
        // Add the component to the canvas
        canvas.appendChild(componentElement);
        
        // Make the component draggable with jsPlumb
        jsPlumbInstance.draggable(instanceId, {
            grid: [10, 10],
            containment: 'canvas'
        });
        
        // Add endpoints to the component
        addEndpoints(instanceId);
        
        // Add the component to the workflow
        workflow.nodes.push({
            id: instanceId,
            type: componentData.id,
            category: componentData.category,
            name: componentData.name,
            params: componentData.params ? { ...componentData.params } : {},
            position: { x, y }
        });
        
        // Add event listeners to the component
        addComponentEventListeners(componentElement);
    }
    
    /**
     * Add jsPlumb endpoints to a component
     * @param {string} instanceId - The component instance ID
     */
    function addEndpoints(instanceId) {
        // Add input endpoint
        jsPlumbInstance.addEndpoint(instanceId, {
            anchor: "Left",
            isTarget: true,
            maxConnections: -1,
            endpoint: ["Dot", { radius: 6 }],
            paintStyle: { fill: "#4a6bff" },
            hoverPaintStyle: { fill: "#3a5bef" }
        });
        
        // Add output endpoint
        jsPlumbInstance.addEndpoint(instanceId, {
            anchor: "Right",
            isSource: true,
            maxConnections: -1,
            endpoint: ["Dot", { radius: 6 }],
            paintStyle: { fill: "#4a6bff" },
            hoverPaintStyle: { fill: "#3a5bef" },
            connectorStyle: { stroke: "#4a6bff", strokeWidth: 2 },
            connectorHoverStyle: { stroke: "#3a5bef", strokeWidth: 3 }
        });
        
        // Handle connection events
        jsPlumbInstance.bind("connection", function(info) {
            // Add the connection to the workflow
            workflow.connections.push({
                source: info.sourceId,
                target: info.targetId
            });
        });
        
        jsPlumbInstance.bind("connectionDetached", function(info) {
            // Remove the connection from the workflow
            const index = workflow.connections.findIndex(
                conn => conn.source === info.sourceId && conn.target === info.targetId
            );
            
            if (index !== -1) {
                workflow.connections.splice(index, 1);
            }
        });
    }
    
    /**
     * Add event listeners to a component
     * @param {HTMLElement} componentElement - The component element
     */
    function addComponentEventListeners(componentElement) {
        // Select component on click
        componentElement.addEventListener('click', function(event) {
            // Prevent click from propagating to canvas
            event.stopPropagation();
            
            // Deselect previously selected component
            if (selectedComponent) {
                selectedComponent.classList.remove('selected');
            }
            
            // Select this component
            componentElement.classList.add('selected');
            selectedComponent = componentElement;
            
            // Show component properties
            showComponentProperties(componentElement);
        });
        
        // Delete component button
        const deleteBtn = componentElement.querySelector('.delete-btn');
        deleteBtn.addEventListener('click', function(event) {
            event.stopPropagation();
            deleteComponent(componentElement.id);
        });
        
        // Configure component button
        const configBtn = componentElement.querySelector('.config-btn');
        configBtn.addEventListener('click', function(event) {
            event.stopPropagation();
            showComponentConfiguration(componentElement);
        });
    }
    
    /**
     * Show component properties in the properties panel
     * @param {HTMLElement} componentElement - The component element
     */
    function showComponentProperties(componentElement) {
        const propertiesContent = document.getElementById('properties-content');
        const componentId = componentElement.getAttribute('data-component-id');
        const componentCategory = componentElement.getAttribute('data-component-category');
        
        // Find the component data
        const componentData = window.mlComponents[componentCategory].find(c => c.id === componentId);
        
        // Find the component instance in the workflow
        const instanceId = componentElement.id;
        const componentInstance = workflow.nodes.find(node => node.id === instanceId);
        
        if (componentData && componentInstance) {
            let propertiesHTML = `
                <div class="property-group">
                    <h4>${componentData.name}</h4>
                    <p>Type: ${componentData.category}</p>
                </div>
            `;
            
            // Add parameters if they exist
            if (componentInstance.params && Object.keys(componentInstance.params).length > 0) {
                propertiesHTML += '<div class="property-group"><h4>Parameters</h4>';
                
                for (const [key, value] of Object.entries(componentInstance.params)) {
                    propertiesHTML += `
                        <div class="property">
                            <label for="${key}-${instanceId}">${key}</label>
                            <input type="text" id="${key}-${instanceId}" value="${value}" 
                                data-param-name="${key}" data-component-id="${instanceId}">
                        </div>
                    `;
                }
                
                propertiesHTML += '</div>';
            }
            
            propertiesContent.innerHTML = propertiesHTML;
            
            // Add event listeners to parameter inputs
            document.querySelectorAll(`[data-component-id="${instanceId}"]`).forEach(input => {
                input.addEventListener('change', function() {
                    const paramName = this.getAttribute('data-param-name');
                    const componentId = this.getAttribute('data-component-id');
                    const value = this.value;
                    
                    // Update the parameter value in the workflow
                    const component = workflow.nodes.find(node => node.id === componentId);
                    if (component && component.params) {
                        component.params[paramName] = value;
                    }
                });
            });
        }
    }
    
    /**
     * Show component configuration modal
     * @param {HTMLElement} componentElement - The component element
     */
    function showComponentConfiguration(componentElement) {
        // Implementation for component configuration modal
        // This would be a more detailed configuration interface than the properties panel
        alert('Configuration modal not implemented yet.');
    }
    
    /**
     * Delete a component from the canvas
     * @param {string} instanceId - The component instance ID
     */
    function deleteComponent(instanceId) {
        // Remove the component from jsPlumb
        jsPlumbInstance.remove(instanceId);
        
        // Remove the component from the workflow
        const index = workflow.nodes.findIndex(node => node.id === instanceId);
        if (index !== -1) {
            workflow.nodes.splice(index, 1);
        }
        
        // Remove any connections involving this component
        workflow.connections = workflow.connections.filter(
            conn => conn.source !== instanceId && conn.target !== instanceId
        );
        
        // Clear properties panel if this was the selected component
        if (selectedComponent && selectedComponent.id === instanceId) {
            selectedComponent = null;
            document.getElementById('properties-content').innerHTML = '<p class="no-selection">No component selected</p>';
        }
    }
    
    /**
     * Initialize event listeners
     */
    function initializeEventListeners() {
        // Category selection
        document.querySelectorAll('.category').forEach(category => {
            category.addEventListener('click', function() {
                const categoryName = this.getAttribute('data-category');
                displayComponentsByCategory(categoryName);
            });
        });
        
        // Search components
        document.getElementById('search-components').addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            searchComponents(searchTerm);
        });
        
        // Run workflow button
        document.getElementById('run-workflow').addEventListener('click', function() {
            if (!isExecuting) {
                executeWorkflow();
            }
        });
        
        // Stop workflow button
        document.getElementById('stop-workflow').addEventListener('click', function() {
            if (isExecuting) {
                stopExecution();
            }
        });
        
        // New project button
        document.getElementById('new-project').addEventListener('click', function() {
            if (confirm('Are you sure you want to create a new project? All unsaved changes will be lost.')) {
                clearWorkspace();
            }
        });
        
        // Save project button
        document.getElementById('save-project').addEventListener('click', function() {
            saveProject();
        });
        
        // Load project button
        document.getElementById('load-project').addEventListener('click', function() {
            // This would typically open a file dialog
            alert('Load project functionality not implemented yet.');
        });
        
        // Export project button
        document.getElementById('export-project').addEventListener('click', function() {
            exportProject();
        });
        
        // Settings button
        document.getElementById('settings').addEventListener('click', function() {
            showSettingsModal();
        });
        
        // Help button
        document.getElementById('help').addEventListener('click', function() {
            // This would typically show a help modal or documentation
            alert('Help functionality not implemented yet.');
        });
        
        // Canvas click (deselect components)
        document.getElementById('canvas').addEventListener('click', function() {
            if (selectedComponent) {
                selectedComponent.classList.remove('selected');
                selectedComponent = null;
                document.getElementById('properties-content').innerHTML = '<p class="no-selection">No component selected</p>';
            }
        });
        
        // Modal close buttons
        document.querySelectorAll('.close').forEach(closeBtn => {
            closeBtn.addEventListener('click', function() {
                this.closest('.modal').style.display = 'none';
            });
        });
        
        // Save settings button
        document.getElementById('save-settings').addEventListener('click', function() {
            saveSettings();
            document.getElementById('settings-modal').style.display = 'none';
        });
        
        // Close modals when clicking outside
        window.addEventListener('click', function(event) {
            document.querySelectorAll('.modal').forEach(modal => {
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            });
        });
    }
    
    /**
     * Initialize socket events
     */
    function initializeSocketEvents() {
        socket.on('connect', function() {
            console.log('Connected to server');
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from server');
        });
        
        socket.on('execution_progress', function(data) {
            updateExecutionProgress(data);
        });
        
        socket.on('execution_complete', function(data) {
            executionComplete(data);
        });
        
        socket.on('execution_error', function(data) {
            executionError(data);
        });
    }
    
    /**
     * Search components by name
     * @param {string} searchTerm - The search term
     */
    function searchComponents(searchTerm) {
        if (!searchTerm) {
            // If search term is empty, show the current category
            const activeCategory = document.querySelector('.category.active');
            if (activeCategory) {
                displayComponentsByCategory(activeCategory.getAttribute('data-category'));
            }
            return;
        }
        
        const componentsContainer = document.getElementById('components-container');
        componentsContainer.innerHTML = '';
        
        let results = [];
        
        // Search in all categories
        for (const [category, components] of Object.entries(window.mlComponents)) {
            const matchingComponents = components.filter(component => 
                component.name.toLowerCase().includes(searchTerm) || 
                component.id.toLowerCase().includes(searchTerm)
            );
            
            results = results.concat(matchingComponents.map(component => ({
                ...component,
                category
            })));
        }
        
        if (results.length === 0) {
            componentsContainer.innerHTML = '<p>No components found matching your search.</p>';
            return;
        }
        
        // Display search results
        results.forEach(component => {
            const componentElement = document.createElement('div');
            componentElement.className = 'component-item';
            componentElement.setAttribute('data-component-id', component.id);
            componentElement.setAttribute('data-component-category', component.category);
            componentElement.setAttribute('draggable', 'true');
            
            componentElement.innerHTML = `
                <h4>${component.name}</h4>
                <p>${component.category}</p>
            `;
            
            componentsContainer.appendChild(componentElement);
        });
    }
    
    /**
     * Execute the current workflow
     */
    function executeWorkflow() {
        if (workflow.nodes.length === 0) {
            showError('Cannot execute an empty workflow. Add some components first.');
            return;
        }
        
        // Set execution state
        isExecuting = true;
        document.getElementById('run-workflow').disabled = true;
        document.getElementById('stop-workflow').disabled = false;
        document.getElementById('execution-status').textContent = 'Running';
        document.getElementById('execution-stats').style.display = 'block';
        
        // Send the workflow to the server for execution
        fetch('/api/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(workflow)
        })
        .then(response => response.json())
        .then(data => {
            executionComplete(data);
        })
        .catch(error => {
            executionError({ message: error.message });
        });
    }
    
    /**
     * Stop the current execution
     */
    function stopExecution() {
        // In a real implementation, this would send a stop signal to the server
        // For now, we'll just reset the UI
        isExecuting = false;
        document.getElementById('run-workflow').disabled = false;
        document.getElementById('stop-workflow').disabled = true;
        document.getElementById('execution-status').textContent = 'Stopped';
    }
    
    /**
     * Update execution progress
     * @param {Object} data - Progress data from the server
     */
    function updateExecutionProgress(data) {
        // Update progress indicators and charts
        document.getElementById('execution-time').textContent = `${data.elapsed_time.toFixed(2)}s`;
        document.getElementById('memory-usage').textContent = `${data.memory_usage.toFixed(2)} MB`;
        
        // Update charts if they exist
        if (window.accuracyChart && data.metrics && data.metrics.accuracy) {
            window.accuracyChart.data.labels.push(data.step);
            window.accuracyChart.data.datasets[0].data.push(data.metrics.accuracy);
            window.accuracyChart.update();
        }
        
        if (window.lossChart && data.metrics && data.metrics.loss) {
            window.lossChart.data.labels.push(data.step);
            window.lossChart.data.datasets[0].data.push(data.metrics.loss);
            window.lossChart.update();
        }
    }
    
    /**
     * Handle execution completion
     * @param {Object} data - Completion data from the server
     */
    function executionComplete(data) {
        isExecuting = false;
        document.getElementById('run-workflow').disabled = false;
        document.getElementById('stop-workflow').disabled = true;
        document.getElementById('execution-status').textContent = 'Completed';
        document.getElementById('execution-time').textContent = `${data.execution_time.toFixed(2)}s`;
        
        // Display final metrics
        if (data.metrics) {
            // Initialize charts if they don't exist
            initializeCharts(data.metrics);
        }
    }
    
    /**
     * Handle execution error
     * @param {Object} data - Error data from the server
     */
    function executionError(data) {
        isExecuting = false;
        document.getElementById('run-workflow').disabled = false;
        document.getElementById('stop-workflow').disabled = true;
        document.getElementById('execution-status').textContent = 'Error';
        
        showError(`Execution failed: ${data.message}`);
    }
    
    /**
     * Initialize charts for displaying metrics
     * @param {Object} metrics - Metrics data
     */
    function initializeCharts(metrics) {
        const accuracyCtx = document.getElementById('accuracy-chart').getContext('2d');
        const lossCtx = document.getElementById('loss-chart').getContext('2d');
        
        // Destroy existing charts if they exist
        if (window.accuracyChart) {
            window.accuracyChart.destroy();
        }
        
        if (window.lossChart) {
            window.lossChart.destroy();
        }
        
        // Create new charts
        window.accuracyChart = new Chart(accuracyCtx, {
            type: 'line',
            data: {
                labels: ['Final'],
                datasets: [{
                    label: 'Accuracy',
                    data: [metrics.accuracy],
                    borderColor: '#28a745',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Accuracy'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
        
        window.lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: ['Final'],
                datasets: [{
                    label: 'Loss',
                    data: [metrics.loss],
                    borderColor: '#dc3545',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    /**
     * Clear the workspace
     */
    function clearWorkspace() {
        // Remove all components from jsPlumb
        jsPlumbInstance.reset();
        
        // Clear the canvas
        document.getElementById('canvas').innerHTML = '';
        
        // Reset the workflow
        workflow = {
            nodes: [],
            connections: []
        };
        
        // Reset component counter
        componentCounter = 0;
        
        // Clear properties panel
        document.getElementById('properties-content').innerHTML = '<p class="no-selection">No component selected</p>';
        
        // Reset selected component
        selectedComponent = null;
        
        // Hide execution stats
        document.getElementById('execution-stats').style.display = 'none';
    }
    
    /**
     * Save the current project
     */
    function saveProject() {
        // In a real implementation, this would save to a server or local file
        const projectData = {
            workflow: workflow,
            version: '1.0'
        };
        
        const dataStr = JSON.stringify(projectData);
        const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
        
        const exportName = 'ml_sandbox_project_' + new Date().toISOString().slice(0, 10) + '.json';
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportName);
        linkElement.click();
    }
    
    /**
     * Export the current project
     */
    function exportProject() {
        // In a real implementation, this would export to various formats (Python code, etc.)
        alert('Export functionality not implemented yet.');
    }
    
    /**
     * Show settings modal
     */
    function showSettingsModal() {
        // Populate settings form with current values
        document.getElementById('theme-select').value = settings.theme;
        document.getElementById('autosave-interval').value = settings.autosaveInterval;
        document.getElementById('execution-timeout').value = settings.executionTimeout;
        document.getElementById('show-tooltips').checked = settings.showTooltips;
        document.getElementById('auto-connect').checked = settings.autoConnect;
        
        // Show the modal
        document.getElementById('settings-modal').style.display = 'block';
    }
    
    /**
     * Save settings
     */
    function saveSettings() {
        settings = {
            theme: document.getElementById('theme-select').value,
            autosaveInterval: parseInt(document.getElementById('autosave-interval').value),
            executionTimeout: parseInt(document.getElementById('execution-timeout').value),
            showTooltips: document.getElementById('show-tooltips').checked,
            autoConnect: document.getElementById('auto-connect').checked
        };
        
        // Save to localStorage
        localStorage.setItem('ml_sandbox_settings', JSON.stringify(settings));
        
        // Apply settings
        applySettings();
    }
    
    /**
     * Load settings from localStorage
     * @returns {Object} The loaded settings or default settings
     */
    function loadSettings() {
        const savedSettings = localStorage.getItem('ml_sandbox_settings');
        
        if (savedSettings) {
            return JSON.parse(savedSettings);
        }
        
        // Default settings
        return {
            theme: 'light',
            autosaveInterval: 5,
            executionTimeout: 300,
            showTooltips: true,
            autoConnect: true
        };
    }
    
    /**
     * Apply settings to the UI
     */
    function applySettings() {
        // Apply theme
        document.body.className = settings.theme;
        
        // Apply other settings as needed
        if (settings.showTooltips) {
            // Enable tooltips
        } else {
            // Disable tooltips
        }
        
        if (settings.autoConnect) {
            // Enable auto-connect functionality
        } else {
            // Disable auto-connect functionality
        }
    }
    
    /**
     * Show an error message
     * @param {string} message - The error message
     */
    function showError(message) {
        const errorModal = document.getElementById('error-modal');
        const errorMessage = document.getElementById('error-message');
        
        errorMessage.textContent = message;
        errorModal.style.display = 'block';
    }
});
