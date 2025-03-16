/**
 * ML Sandbox - UI JavaScript
 * Handles user interface interactions and visual elements
 */

class UI {
    /**
     * Initialize the UI
     * @param {Object} options - UI options
     */
    constructor(options = {}) {
        this.options = {
            theme: 'light',
            showTooltips: true,
            autoConnect: true,
            ...options
        };
        
        // DOM elements
        this.elements = {
            canvas: document.getElementById('canvas'),
            componentsContainer: document.getElementById('components-container'),
            propertiesContent: document.getElementById('properties-content'),
            searchInput: document.getElementById('search-components'),
            categories: document.querySelectorAll('.category'),
            runButton: document.getElementById('run-workflow'),
            stopButton: document.getElementById('stop-workflow'),
            newButton: document.getElementById('new-project'),
            saveButton: document.getElementById('save-project'),
            loadButton: document.getElementById('load-project'),
            exportButton: document.getElementById('export-project'),
            settingsButton: document.getElementById('settings'),
            helpButton: document.getElementById('help'),
            executionStats: document.getElementById('execution-stats'),
            executionStatus: document.getElementById('execution-status'),
            executionTime: document.getElementById('execution-time'),
            memoryUsage: document.getElementById('memory-usage')
        };
        
        // State
        this.selectedComponent = null;
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        
        // Initialize
        this.initializeEventListeners();
        this.applyTheme(this.options.theme);
    }
    
    /**
     * Initialize event listeners
     */
    initializeEventListeners() {
        // Category selection
        this.elements.categories.forEach(category => {
            category.addEventListener('click', () => {
                const categoryName = category.getAttribute('data-category');
                this.selectCategory(categoryName);
                
                // Trigger custom event
                const event = new CustomEvent('category-selected', {
                    detail: { category: categoryName }
                });
                document.dispatchEvent(event);
            });
        });
        
        // Component search
        this.elements.searchInput.addEventListener('input', () => {
            const searchTerm = this.elements.searchInput.value.toLowerCase();
            
            // Trigger custom event
            const event = new CustomEvent('search-components', {
                detail: { searchTerm }
            });
            document.dispatchEvent(event);
        });
        
        // Canvas click (deselect components)
        this.elements.canvas.addEventListener('click', (event) => {
            if (event.target === this.elements.canvas) {
                this.deselectComponent();
                
                // Trigger custom event
                const customEvent = new CustomEvent('canvas-clicked');
                document.dispatchEvent(customEvent);
            }
        });
        
        // Modal close buttons
        document.querySelectorAll('.close').forEach(closeBtn => {
            closeBtn.addEventListener('click', () => {
                closeBtn.closest('.modal').style.display = 'none';
            });
        });
        
        // Close modals when clicking outside
        window.addEventListener('click', (event) => {
            document.querySelectorAll('.modal').forEach(modal => {
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            });
        });
        
        // Save settings button
        document.getElementById('save-settings').addEventListener('click', () => {
            this.saveSettings();
            document.getElementById('settings-modal').style.display = 'none';
        });
        
        // Theme selection
        document.getElementById('theme-select').addEventListener('change', (event) => {
            this.applyTheme(event.target.value);
        });
    }
    
    /**
     * Select a category
     * @param {string} categoryName - The category name
     */
    selectCategory(categoryName) {
        // Update active category
        this.elements.categories.forEach(category => {
            category.classList.remove('active');
            if (category.getAttribute('data-category') === categoryName) {
                category.classList.add('active');
            }
        });
    }
    
    /**
     * Display components by category
     * @param {Array} components - The components to display
     */
    displayComponents(components) {
        this.elements.componentsContainer.innerHTML = '';
        
        if (!components || components.length === 0) {
            this.elements.componentsContainer.innerHTML = '<p>No components found.</p>';
            return;
        }
        
        components.forEach(component => {
            const componentElement = document.createElement('div');
            componentElement.className = 'component-item';
            componentElement.setAttribute('data-component-id', component.id);
            componentElement.setAttribute('data-component-category', component.category);
            componentElement.setAttribute('draggable', 'true');
            
            componentElement.innerHTML = `
                <h4>${component.name}</h4>
                <p>${component.category}</p>
            `;
            
            this.elements.componentsContainer.appendChild(componentElement);
            
            // Add drag event listeners
            this.addDragListeners(componentElement);
        });
    }
    
    /**
     * Add drag event listeners to a component
     * @param {HTMLElement} element - The component element
     */
    addDragListeners(element) {
        element.addEventListener('dragstart', (event) => {
            event.dataTransfer.setData('component-id', element.getAttribute('data-component-id'));
            event.dataTransfer.setData('component-category', element.getAttribute('data-component-category'));
        });
    }
    
    /**
     * Create a component on the canvas
     * @param {Object} componentData - The component data
     * @param {number} x - The X position
     * @param {number} y - The Y position
     * @returns {HTMLElement} The created component element
     */
    createCanvasComponent(componentData, x, y) {
        const componentElement = document.createElement('div');
        componentElement.className = 'canvas-component';
        componentElement.id = componentData.instanceId;
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
        this.elements.canvas.appendChild(componentElement);
        
        // Add event listeners
        this.addComponentEventListeners(componentElement);
        
        return componentElement;
    }
    
    /**
     * Add event listeners to a canvas component
     * @param {HTMLElement} componentElement - The component element
     */
    addComponentEventListeners(componentElement) {
        // Select component on click
        componentElement.addEventListener('click', (event) => {
            // Prevent click from propagating to canvas
            event.stopPropagation();
            
            this.selectComponent(componentElement);
        });
        
        // Delete component button
        const deleteBtn = componentElement.querySelector('.delete-btn');
        deleteBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            
            // Trigger custom event
            const customEvent = new CustomEvent('delete-component', {
                detail: { componentId: componentElement.id }
            });
            document.dispatchEvent(customEvent);
        });
        
        // Configure component button
        const configBtn = componentElement.querySelector('.config-btn');
        configBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            
            // Trigger custom event
            const customEvent = new CustomEvent('configure-component', {
                detail: { componentId: componentElement.id }
            });
            document.dispatchEvent(customEvent);
        });
    }
    
    /**
     * Select a component
     * @param {HTMLElement} componentElement - The component element
     */
    selectComponent(componentElement) {
        // Deselect previously selected component
        if (this.selectedComponent) {
            this.selectedComponent.classList.remove('selected');
        }
        
        // Select this component
        componentElement.classList.add('selected');
        this.selectedComponent = componentElement;
        
        // Trigger custom event
        const event = new CustomEvent('component-selected', {
            detail: { componentId: componentElement.id }
        });
        document.dispatchEvent(event);
    }
    
    /**
     * Deselect the currently selected component
     */
    deselectComponent() {
        if (this.selectedComponent) {
            this.selectedComponent.classList.remove('selected');
            this.selectedComponent = null;
            
            // Clear properties panel
            this.elements.propertiesContent.innerHTML = '<p class="no-selection">No component selected</p>';
            
            // Trigger custom event
            const event = new CustomEvent('component-deselected');
            document.dispatchEvent(event);
        }
    }
    
    /**
     * Show component properties in the properties panel
     * @param {Object} componentData - The component data
     */
    showComponentProperties(componentData) {
        let propertiesHTML = `
            <div class="property-group">
                <h4>${componentData.name}</h4>
                <p>Type: ${componentData.category}</p>
            </div>
        `;
        
        // Add parameters if they exist
        if (componentData.params && Object.keys(componentData.params).length > 0) {
            propertiesHTML += '<div class="property-group"><h4>Parameters</h4>';
            
            for (const [key, value] of Object.entries(componentData.params)) {
                propertiesHTML += `
                    <div class="property">
                        <label for="${key}-${componentData.instanceId}">${key}</label>
                        <input type="text" id="${key}-${componentData.instanceId}" value="${value}" 
                            data-param-name="${key}" data-component-id="${componentData.instanceId}">
                    </div>
                `;
            }
            
            propertiesHTML += '</div>';
        }
        
        this.elements.propertiesContent.innerHTML = propertiesHTML;
        
        // Add event listeners to parameter inputs
        document.querySelectorAll(`[data-component-id="${componentData.instanceId}"]`).forEach(input => {
            input.addEventListener('change', () => {
                const paramName = input.getAttribute('data-param-name');
                const componentId = input.getAttribute('data-component-id');
                const value = input.value;
                
                // Trigger custom event
                const event = new CustomEvent('param-changed', {
                    detail: { componentId, paramName, value }
                });
                document.dispatchEvent(event);
            });
        });
    }
    
    /**
     * Update execution status
     * @param {Object} data - Execution status data
     */
    updateExecutionStatus(data) {
        this.elements.executionStatus.textContent = data.status;
        this.elements.executionTime.textContent = `${data.time.toFixed(2)}s`;
        this.elements.memoryUsage.textContent = `${data.memory.toFixed(2)} MB`;
        
        if (data.status === 'running') {
            this.elements.runButton.disabled = true;
            this.elements.stopButton.disabled = false;
        } else {
            this.elements.runButton.disabled = false;
            this.elements.stopButton.disabled = true;
        }
        
        this.elements.executionStats.style.display = 'block';
    }
    
    /**
     * Show an error message
     * @param {string} message - The error message
     */
    showError(message) {
        const errorModal = document.getElementById('error-modal');
        const errorMessage = document.getElementById('error-message');
        
        errorMessage.textContent = message;
        errorModal.style.display = 'block';
    }
    
    /**
     * Show settings modal
     * @param {Object} settings - Current settings
     */
    showSettings(settings) {
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
     * @returns {Object} The saved settings
     */
    saveSettings() {
        const settings = {
            theme: document.getElementById('theme-select').value,
            autosaveInterval: parseInt(document.getElementById('autosave-interval').value),
            executionTimeout: parseInt(document.getElementById('execution-timeout').value),
            showTooltips: document.getElementById('show-tooltips').checked,
            autoConnect: document.getElementById('auto-connect').checked
        };
        
        // Apply settings
        this.applyTheme(settings.theme);
        this.options = { ...this.options, ...settings };
        
        // Trigger custom event
        const event = new CustomEvent('settings-saved', {
            detail: { settings }
        });
        document.dispatchEvent(event);
        
        return settings;
    }
    
    /**
     * Apply theme
     * @param {string} theme - The theme name
     */
    applyTheme(theme) {
        document.body.className = theme;
    }
}

// Export the UI class
window.UI = UI;
