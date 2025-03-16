/**
 * ML Sandbox - Workflow JavaScript
 * Handles ML workflow execution and management
 */

class MLWorkflow {
    /**
     * Create a new ML workflow
     */
    constructor() {
        this.nodes = [];
        this.connections = [];
        this.executionOrder = [];
        this.executionResults = {};
        this.isExecuting = false;
        this.executionProgress = 0;
        this.executionStartTime = null;
        this.executionEndTime = null;
        this.executionError = null;
        this.onProgressCallback = null;
        this.onCompleteCallback = null;
        this.onErrorCallback = null;
    }

    /**
     * Add a node to the workflow
     * @param {MLComponent} node - The node to add
     */
    addNode(node) {
        this.nodes.push(node);
    }

    /**
     * Remove a node from the workflow
     * @param {string} nodeId - The ID of the node to remove
     */
    removeNode(nodeId) {
        // Remove the node
        this.nodes = this.nodes.filter(node => node.instanceId !== nodeId);
        
        // Remove connections involving this node
        this.connections = this.connections.filter(
            conn => conn.source !== nodeId && conn.target !== nodeId
        );
    }

    /**
     * Add a connection between nodes
     * @param {string} sourceId - Source node ID
     * @param {string} targetId - Target node ID
     * @param {string} outputName - Output port name
     * @param {string} inputName - Input port name
     */
    addConnection(sourceId, targetId, outputName = 'output', inputName = 'input') {
        this.connections.push({
            source: sourceId,
            target: targetId,
            outputName,
            inputName
        });
    }

    /**
     * Remove a connection
     * @param {string} sourceId - Source node ID
     * @param {string} targetId - Target node ID
     * @param {string} outputName - Output port name
     * @param {string} inputName - Input port name
     */
    removeConnection(sourceId, targetId, outputName = 'output', inputName = 'input') {
        this.connections = this.connections.filter(
            conn => !(
                conn.source === sourceId &&
                conn.target === targetId &&
                conn.outputName === outputName &&
                conn.inputName === inputName
            )
        );
    }

    /**
     * Determine the execution order of nodes
     * @returns {Array} Ordered array of node IDs
     */
    determineExecutionOrder() {
        // Reset execution order
        this.executionOrder = [];
        
        // Find nodes with no inputs (starting nodes)
        const startingNodes = this.nodes.filter(node => {
            return !this.connections.some(conn => conn.target === node.instanceId);
        });
        
        // If there are no starting nodes, we can't execute
        if (startingNodes.length === 0) {
            throw new Error('No starting nodes found in the workflow');
        }
        
        // Create a queue for topological sort
        const queue = [...startingNodes];
        const visited = new Set();
        
        // Process the queue
        while (queue.length > 0) {
            const node = queue.shift();
            
            // Skip if already visited
            if (visited.has(node.instanceId)) {
                continue;
            }
            
            // Mark as visited
            visited.add(node.instanceId);
            this.executionOrder.push(node.instanceId);
            
            // Find all connections from this node
            const outgoingConnections = this.connections.filter(
                conn => conn.source === node.instanceId
            );
            
            // Add target nodes to the queue
            for (const conn of outgoingConnections) {
                const targetNode = this.nodes.find(node => node.instanceId === conn.target);
                
                if (targetNode) {
                    // Check if all inputs to this target node have been visited
                    const incomingConnections = this.connections.filter(
                        c => c.target === targetNode.instanceId
                    );
                    
                    const allInputsVisited = incomingConnections.every(
                        c => visited.has(c.source)
                    );
                    
                    if (allInputsVisited) {
                        queue.push(targetNode);
                    }
                }
            }
        }
        
        // Check if all nodes are in the execution order
        if (this.executionOrder.length !== this.nodes.length) {
            throw new Error('Circular dependency detected in the workflow');
        }
        
        return this.executionOrder;
    }

    /**
     * Validate the workflow
     * @returns {Object} Validation result
     */
    validate() {
        const errors = [];
        
        // Check for empty workflow
        if (this.nodes.length === 0) {
            errors.push('Workflow is empty');
            return { valid: false, errors };
        }
        
        // Check for disconnected nodes
        for (const node of this.nodes) {
            // Skip data source nodes (they don't need inputs)
            if (node.category === 'data') {
                continue;
            }
            
            // Check if the node has any inputs
            const hasInputs = this.connections.some(conn => conn.target === node.instanceId);
            
            if (!hasInputs) {
                errors.push(`Node "${node.name}" (${node.instanceId}) has no inputs`);
            }
        }
        
        // Check for parameter validation
        for (const node of this.nodes) {
            if (!node.validateParams()) {
                errors.push(`Invalid parameters for node "${node.name}" (${node.instanceId})`);
            }
        }
        
        // Try to determine execution order
        try {
            this.determineExecutionOrder();
        } catch (error) {
            errors.push(error.message);
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    }

    /**
     * Execute the workflow
     * @param {Function} onProgress - Progress callback
     * @param {Function} onComplete - Completion callback
     * @param {Function} onError - Error callback
     */
    async execute(onProgress, onComplete, onError) {
        // Set callbacks
        this.onProgressCallback = onProgress;
        this.onCompleteCallback = onComplete;
        this.onErrorCallback = onError;
        
        // Reset execution state
        this.isExecuting = true;
        this.executionProgress = 0;
        this.executionStartTime = Date.now();
        this.executionResults = {};
        this.executionError = null;
        
        // Validate the workflow
        const validation = this.validate();
        
        if (!validation.valid) {
            this.handleError(new Error(`Workflow validation failed: ${validation.errors.join(', ')}`));
            return;
        }
        
        try {
            // Determine execution order
            const executionOrder = this.determineExecutionOrder();
            
            // Execute each node in order
            for (let i = 0; i < executionOrder.length; i++) {
                const nodeId = executionOrder[i];
                const node = this.nodes.find(n => n.instanceId === nodeId);
                
                if (!node) {
                    throw new Error(`Node not found: ${nodeId}`);
                }
                
                // Update progress
                this.executionProgress = (i / executionOrder.length) * 100;
                this.notifyProgress();
                
                // Get input data for this node
                const inputs = this.getNodeInputs(nodeId);
                
                // Execute the node
                const outputs = await node.execute(inputs);
                
                // Store the results
                this.executionResults[nodeId] = outputs;
            }
            
            // Execution complete
            this.executionProgress = 100;
            this.executionEndTime = Date.now();
            this.isExecuting = false;
            
            // Notify completion
            this.notifyComplete();
        } catch (error) {
            this.handleError(error);
        }
    }

    /**
     * Get input data for a node
     * @param {string} nodeId - The node ID
     * @returns {Object} Input data
     */
    getNodeInputs(nodeId) {
        const inputs = {};
        
        // Find all connections to this node
        const incomingConnections = this.connections.filter(
            conn => conn.target === nodeId
        );
        
        // Get the data from each source node
        for (const conn of incomingConnections) {
            const sourceResults = this.executionResults[conn.source];
            
            if (!sourceResults) {
                throw new Error(`No results available from source node: ${conn.source}`);
            }
            
            const outputData = sourceResults[conn.outputName];
            
            if (outputData === undefined) {
                throw new Error(`Output "${conn.outputName}" not found in source node: ${conn.source}`);
            }
            
            inputs[conn.inputName] = outputData;
        }
        
        return inputs;
    }

    /**
     * Stop the workflow execution
     */
    stop() {
        if (this.isExecuting) {
            this.isExecuting = false;
            this.executionEndTime = Date.now();
            this.notifyComplete();
        }
    }

    /**
     * Notify progress
     */
    notifyProgress() {
        if (this.onProgressCallback) {
            const elapsedTime = (Date.now() - this.executionStartTime) / 1000;
            
            this.onProgressCallback({
                progress: this.executionProgress,
                elapsedTime,
                currentNode: this.executionOrder[Math.floor((this.executionProgress / 100) * this.executionOrder.length)]
            });
        }
    }

    /**
     * Notify completion
     */
    notifyComplete() {
        if (this.onCompleteCallback) {
            const executionTime = (this.executionEndTime - this.executionStartTime) / 1000;
            
            this.onCompleteCallback({
                executionTime,
                results: this.executionResults
            });
        }
    }

    /**
     * Handle error
     * @param {Error} error - The error
     */
    handleError(error) {
        this.isExecuting = false;
        this.executionEndTime = Date.now();
        this.executionError = error;
        
        if (this.onErrorCallback) {
            this.onErrorCallback({
                message: error.message,
                stack: error.stack
            });
        }
    }

    /**
     * Export the workflow as JSON
     * @returns {Object} Workflow JSON
     */
    exportJson() {
        return {
            nodes: this.nodes.map(node => ({
                id: node.instanceId,
                type: node.id,
                category: node.category,
                name: node.name,
                params: { ...node.params },
                position: { ...node.position }
            })),
            connections: this.connections.map(conn => ({
                source: conn.source,
                target: conn.target,
                outputName: conn.outputName,
                inputName: conn.inputName
            }))
        };
    }

    /**
     * Import a workflow from JSON
     * @param {Object} json - Workflow JSON
     */
    importJson(json) {
        // Clear current workflow
        this.nodes = [];
        this.connections = [];
        this.executionOrder = [];
        this.executionResults = {};
        
        // Import nodes
        for (const nodeData of json.nodes) {
            const node = ComponentFactory.createComponent(
                nodeData.type,
                nodeData.category,
                nodeData.id,
                nodeData.params,
                nodeData.position
            );
            
            this.nodes.push(node);
        }
        
        // Import connections
        for (const connData of json.connections) {
            this.connections.push({
                source: connData.source,
                target: connData.target,
                outputName: connData.outputName || 'output',
                inputName: connData.inputName || 'input'
            });
        }
    }

    /**
     * Export the workflow as Python code
     * @returns {string} Python code
     */
    exportPythonCode() {
        let code = '# Generated by ML Sandbox\n';
        code += 'import numpy as np\n';
        code += 'import tensorflow as tf\n';
        code += 'from tensorflow.keras import layers, models, optimizers, losses, metrics\n\n';
        
        // Generate model building code
        code += '# Build the model\n';
        code += 'def build_model():\n';
        
        // Find input nodes (data loaders)
        const inputNodes = this.nodes.filter(node => node.category === 'data');
        
        if (inputNodes.length === 0) {
            code += '    # No input nodes found\n';
            code += '    return None\n\n';
            return code;
        }
        
        // Start with a sequential model for simplicity
        code += '    model = models.Sequential()\n\n';
        
        // Add layer nodes in execution order
        const layerNodes = this.nodes.filter(node => node.category === 'layer');
        
        for (const node of layerNodes) {
            code += `    # ${node.name}\n`;
            
            switch (node.id) {
                case 'dense':
                    code += `    model.add(layers.Dense(${node.params.units}, activation='${node.params.activation}'))\n`;
                    break;
                case 'conv2d':
                    code += `    model.add(layers.Conv2D(${node.params.filters}, (${node.params.kernel_size}, ${node.params.kernel_size}), activation='${node.params.activation}'))\n`;
                    break;
                case 'maxpool2d':
                    code += `    model.add(layers.MaxPooling2D(pool_size=(${node.params.pool_size}, ${node.params.pool_size})))\n`;
                    break;
                case 'dropout':
                    code += `    model.add(layers.Dropout(${node.params.rate}))\n`;
                    break;
                case 'flatten':
                    code += '    model.add(layers.Flatten())\n';
                    break;
                default:
                    code += `    # Unsupported layer: ${node.id}\n`;
            }
        }
        
        code += '\n    return model\n\n';
        
        // Generate training code
        code += '# Train the model\n';
        code += 'def train_model(model, x_train, y_train, epochs=10, batch_size=32):\n';
        
        // Find optimizer and loss nodes
        const optimizerNode = this.nodes.find(node => node.category === 'optimizer');
        const lossNode = this.nodes.find(node => node.category === 'loss');
        
        if (optimizerNode && lossNode) {
            code += `    # Compile the model with ${optimizerNode.name} optimizer and ${lossNode.name}\n`;
            
            let optimizerCode;
            switch (optimizerNode.id) {
                case 'adam':
                    optimizerCode = `optimizers.Adam(learning_rate=${optimizerNode.params.learning_rate})`;
                    break;
                case 'sgd':
                    optimizerCode = `optimizers.SGD(learning_rate=${optimizerNode.params.learning_rate})`;
                    break;
                case 'rmsprop':
                    optimizerCode = `optimizers.RMSprop(learning_rate=${optimizerNode.params.learning_rate})`;
                    break;
                default:
                    optimizerCode = 'optimizers.Adam()';
            }
            
            let lossCode;
            switch (lossNode.id) {
                case 'categorical_crossentropy':
                    lossCode = 'losses.categorical_crossentropy';
                    break;
                case 'binary_crossentropy':
                    lossCode = 'losses.binary_crossentropy';
                    break;
                case 'mse':
                    lossCode = 'losses.mean_squared_error';
                    break;
                default:
                    lossCode = 'losses.categorical_crossentropy';
            }
            
            code += `    model.compile(optimizer=${optimizerCode}, loss=${lossCode}, metrics=['accuracy'])\n\n`;
        } else {
            code += '    # Compile the model with default settings\n';
            code += "    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n\n";
        }
        
        code += '    # Train the model\n';
        code += '    history = model.fit(\n';
        code += '        x_train, y_train,\n';
        code += '        epochs=epochs,\n';
        code += '        batch_size=batch_size,\n';
        code += '        validation_split=0.2\n';
        code += '    )\n\n';
        code += '    return history\n\n';
        
        // Generate main code
        code += '# Main function\n';
        code += 'def main():\n';
        code += '    # Load data\n';
        code += '    # TODO: Replace with actual data loading code\n';
        code += '    x_train = np.random.random((1000, 28, 28, 1))\n';
        code += '    y_train = np.random.randint(10, size=(1000, 1))\n';
        code += '    y_train = tf.keras.utils.to_categorical(y_train, 10)\n\n';
        code += '    # Build model\n';
        code += '    model = build_model()\n\n';
        code += '    # Train model\n';
        code += '    history = train_model(model, x_train, y_train)\n\n';
        code += '    # Evaluate model\n';
        code += '    # TODO: Add evaluation code\n\n';
        code += '    # Save model\n';
        code += "    model.save('model.h5')\n\n";
        code += '    print(\"Model saved to model.h5\")\n\n';
        code += 'if __name__ == \"__main__\":\n';
        code += '    main()\n';
        
        return code;
    }
}

// Workflow Manager
class WorkflowManager {
    /**
     * Create a new workflow manager
     */
    constructor() {
        this.workflows = {};
        this.activeWorkflowId = null;
    }

    /**
     * Create a new workflow
     * @param {string} id - Workflow ID
     * @returns {MLWorkflow} The created workflow
     */
    createWorkflow(id = null) {
        id = id || `workflow-${Date.now()}`;
        
        if (this.workflows[id]) {
            throw new Error(`Workflow with ID "${id}" already exists`);
        }
        
        this.workflows[id] = new MLWorkflow();
        this.activeWorkflowId = id;
        
        return this.workflows[id];
    }

    /**
     * Get a workflow by ID
     * @param {string} id - Workflow ID
     * @returns {MLWorkflow} The workflow
     */
    getWorkflow(id) {
        if (!this.workflows[id]) {
            throw new Error(`Workflow with ID "${id}" not found`);
        }
        
        return this.workflows[id];
    }

    /**
     * Get the active workflow
     * @returns {MLWorkflow} The active workflow
     */
    getActiveWorkflow() {
        if (!this.activeWorkflowId || !this.workflows[this.activeWorkflowId]) {
            return this.createWorkflow();
        }
        
        return this.workflows[this.activeWorkflowId];
    }

    /**
     * Set the active workflow
     * @param {string} id - Workflow ID
     */
    setActiveWorkflow(id) {
        if (!this.workflows[id]) {
            throw new Error(`Workflow with ID "${id}" not found`);
        }
        
        this.activeWorkflowId = id;
    }

    /**
     * Delete a workflow
     * @param {string} id - Workflow ID
     */
    deleteWorkflow(id) {
        if (!this.workflows[id]) {
            throw new Error(`Workflow with ID "${id}" not found`);
        }
        
        delete this.workflows[id];
        
        if (this.activeWorkflowId === id) {
            this.activeWorkflowId = Object.keys(this.workflows)[0] || null;
        }
    }

    /**
     * Save a workflow to localStorage
     * @param {string} id - Workflow ID
     * @param {string} name - Workflow name
     */
    saveWorkflow(id, name) {
        if (!this.workflows[id]) {
            throw new Error(`Workflow with ID "${id}" not found`);
        }
        
        const workflow = this.workflows[id];
        const savedWorkflows = JSON.parse(localStorage.getItem('ml_sandbox_workflows') || '{}');
        
        savedWorkflows[id] = {
            name,
            data: workflow.exportJson(),
            timestamp: Date.now()
        };
        
        localStorage.setItem('ml_sandbox_workflows', JSON.stringify(savedWorkflows));
    }

    /**
     * Load workflows from localStorage
     */
    loadWorkflows() {
        const savedWorkflows = JSON.parse(localStorage.getItem('ml_sandbox_workflows') || '{}');
        
        for (const [id, data] of Object.entries(savedWorkflows)) {
            const workflow = new MLWorkflow();
            workflow.importJson(data.data);
            this.workflows[id] = workflow;
        }
        
        if (Object.keys(this.workflows).length > 0) {
            this.activeWorkflowId = Object.keys(this.workflows)[0];
        }
    }

    /**
     * Get saved workflow metadata
     * @returns {Array} Array of workflow metadata
     */
    getSavedWorkflowsMetadata() {
        const savedWorkflows = JSON.parse(localStorage.getItem('ml_sandbox_workflows') || '{}');
        
        return Object.entries(savedWorkflows).map(([id, data]) => ({
            id,
            name: data.name,
            timestamp: data.timestamp,
            nodeCount: data.data.nodes.length
        }));
    }
}
