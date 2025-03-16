/**
 * ML Sandbox - Components JavaScript
 * Handles ML component definitions and behavior
 */

class MLComponent {
    /**
     * Create a new ML component
     * @param {Object} config - Component configuration
     */
    constructor(config) {
        this.id = config.id;
        this.instanceId = config.instanceId;
        this.name = config.name;
        this.category = config.category;
        this.params = config.params || {};
        this.inputs = config.inputs || [];
        this.outputs = config.outputs || [];
        this.position = config.position || { x: 0, y: 0 };
        this.state = 'idle'; // idle, running, completed, error
    }

    /**
     * Get component parameters
     * @returns {Object} Component parameters
     */
    getParams() {
        return this.params;
    }

    /**
     * Set component parameters
     * @param {Object} params - Component parameters
     */
    setParams(params) {
        this.params = { ...this.params, ...params };
    }

    /**
     * Validate component parameters
     * @returns {boolean} True if parameters are valid
     */
    validateParams() {
        // Base validation - to be overridden by specific components
        return true;
    }

    /**
     * Execute the component
     * @param {Object} inputs - Input data
     * @returns {Promise<Object>} Output data
     */
    async execute(inputs) {
        // Base execution - to be overridden by specific components
        this.state = 'running';
        
        try {
            // Simulate processing
            await new Promise(resolve => setTimeout(resolve, 500));
            
            this.state = 'completed';
            return { output: inputs };
        } catch (error) {
            this.state = 'error';
            throw error;
        }
    }

    /**
     * Get component description
     * @returns {string} Component description
     */
    getDescription() {
        // Base description - to be overridden by specific components
        return `${this.name} (${this.category})`;
    }

    /**
     * Get component documentation
     * @returns {string} Component documentation
     */
    getDocumentation() {
        // Base documentation - to be overridden by specific components
        return `
            <h3>${this.name}</h3>
            <p>Category: ${this.category}</p>
            <p>This is a base component. Specific components will provide more detailed documentation.</p>
        `;
    }

    /**
     * Get component visual representation
     * @returns {Object} Visual representation data
     */
    getVisualRepresentation() {
        return {
            shape: 'rectangle',
            color: this.getCategoryColor(),
            icon: this.getCategoryIcon()
        };
    }

    /**
     * Get color based on component category
     * @returns {string} Hex color code
     */
    getCategoryColor() {
        const colors = {
            'layer': '#4a6bff',
            'activation': '#28a745',
            'optimizer': '#dc3545',
            'loss': '#ffc107',
            'data': '#17a2b8',
            'preprocessing': '#6f42c1',
            'evaluation': '#fd7e14',
            'visualization': '#20c997'
        };
        
        return colors[this.category] || '#6c757d';
    }

    /**
     * Get icon based on component category
     * @returns {string} FontAwesome icon class
     */
    getCategoryIcon() {
        const icons = {
            'layer': 'fa-layer-group',
            'activation': 'fa-bolt',
            'optimizer': 'fa-sliders-h',
            'loss': 'fa-chart-line',
            'data': 'fa-database',
            'preprocessing': 'fa-filter',
            'evaluation': 'fa-check-circle',
            'visualization': 'fa-chart-bar'
        };
        
        return icons[this.category] || 'fa-cube';
    }
}

// Layer Components
class DenseLayer extends MLComponent {
    constructor(config) {
        super({
            ...config,
            inputs: ['input'],
            outputs: ['output']
        });
    }

    validateParams() {
        return (
            this.params.units > 0 &&
            typeof this.params.activation === 'string'
        );
    }

    async execute(inputs) {
        this.state = 'running';
        
        try {
            // In a real implementation, this would create and execute a dense layer
            await new Promise(resolve => setTimeout(resolve, 300));
            
            this.state = 'completed';
            return {
                output: {
                    shape: [this.params.units],
                    activation: this.params.activation,
                    data: 'tensor_data' // Placeholder for actual tensor data
                }
            };
        } catch (error) {
            this.state = 'error';
            throw error;
        }
    }

    getDescription() {
        return `Dense layer with ${this.params.units} units and ${this.params.activation} activation`;
    }

    getDocumentation() {
        return `
            <h3>Dense Layer</h3>
            <p>A densely-connected neural network layer.</p>
            <h4>Parameters:</h4>
            <ul>
                <li><strong>units:</strong> Number of neurons in the layer</li>
                <li><strong>activation:</strong> Activation function to use</li>
            </ul>
            <h4>Inputs:</h4>
            <ul>
                <li><strong>input:</strong> Input tensor</li>
            </ul>
            <h4>Outputs:</h4>
            <ul>
                <li><strong>output:</strong> Output tensor</li>
            </ul>
        `;
    }
}

class Conv2DLayer extends MLComponent {
    constructor(config) {
        super({
            ...config,
            inputs: ['input'],
            outputs: ['output']
        });
    }

    validateParams() {
        return (
            this.params.filters > 0 &&
            this.params.kernel_size > 0 &&
            typeof this.params.activation === 'string'
        );
    }

    async execute(inputs) {
        this.state = 'running';
        
        try {
            // In a real implementation, this would create and execute a Conv2D layer
            await new Promise(resolve => setTimeout(resolve, 500));
            
            this.state = 'completed';
            return {
                output: {
                    filters: this.params.filters,
                    kernel_size: this.params.kernel_size,
                    activation: this.params.activation,
                    data: 'tensor_data' // Placeholder for actual tensor data
                }
            };
        } catch (error) {
            this.state = 'error';
            throw error;
        }
    }

    getDescription() {
        return `Conv2D layer with ${this.params.filters} filters and ${this.params.kernel_size}x${this.params.kernel_size} kernel`;
    }

    getDocumentation() {
        return `
            <h3>Conv2D Layer</h3>
            <p>2D convolutional layer for spatial convolution over images.</p>
            <h4>Parameters:</h4>
            <ul>
                <li><strong>filters:</strong> Number of output filters</li>
                <li><strong>kernel_size:</strong> Size of the convolution kernel</li>
                <li><strong>activation:</strong> Activation function to use</li>
            </ul>
            <h4>Inputs:</h4>
            <ul>
                <li><strong>input:</strong> Input tensor (4D)</li>
            </ul>
            <h4>Outputs:</h4>
            <ul>
                <li><strong>output:</strong> Output tensor (4D)</li>
            </ul>
        `;
    }
}

// Activation Components
class ActivationComponent extends MLComponent {
    constructor(config) {
        super({
            ...config,
            inputs: ['input'],
            outputs: ['output']
        });
    }

    async execute(inputs) {
        this.state = 'running';
        
        try {
            // In a real implementation, this would apply the activation function
            await new Promise(resolve => setTimeout(resolve, 100));
            
            this.state = 'completed';
            return {
                output: {
                    activation: this.id,
                    data: 'tensor_data' // Placeholder for actual tensor data
                }
            };
        } catch (error) {
            this.state = 'error';
            throw error;
        }
    }

    getDescription() {
        return `${this.name} activation function`;
    }

    getDocumentation() {
        let description = '';
        
        switch (this.id) {
            case 'relu':
                description = 'Applies the rectified linear unit activation function.';
                break;
            case 'sigmoid':
                description = 'Applies the sigmoid activation function.';
                break;
            case 'tanh':
                description = 'Applies the hyperbolic tangent activation function.';
                break;
            case 'softmax':
                description = 'Applies the softmax activation function.';
                break;
            default:
                description = 'Applies an activation function.';
        }
        
        return `
            <h3>${this.name} Activation</h3>
            <p>${description}</p>
            <h4>Inputs:</h4>
            <ul>
                <li><strong>input:</strong> Input tensor</li>
            </ul>
            <h4>Outputs:</h4>
            <ul>
                <li><strong>output:</strong> Output tensor</li>
            </ul>
        `;
    }
}

// Optimizer Components
class OptimizerComponent extends MLComponent {
    constructor(config) {
        super({
            ...config,
            inputs: ['model', 'loss'],
            outputs: ['optimized_model']
        });
    }

    validateParams() {
        return this.params.learning_rate > 0;
    }

    async execute(inputs) {
        this.state = 'running';
        
        try {
            // In a real implementation, this would configure an optimizer
            await new Promise(resolve => setTimeout(resolve, 200));
            
            this.state = 'completed';
            return {
                optimized_model: {
                    optimizer: this.id,
                    learning_rate: this.params.learning_rate,
                    model: inputs.model,
                    loss: inputs.loss
                }
            };
        } catch (error) {
            this.state = 'error';
            throw error;
        }
    }

    getDescription() {
        return `${this.name} optimizer with learning rate ${this.params.learning_rate}`;
    }

    getDocumentation() {
        let description = '';
        
        switch (this.id) {
            case 'adam':
                description = 'Adam optimizer - Adaptive Moment Estimation.';
                break;
            case 'sgd':
                description = 'Stochastic Gradient Descent optimizer.';
                break;
            case 'rmsprop':
                description = 'RMSprop optimizer - Root Mean Square Propagation.';
                break;
            default:
                description = 'An optimization algorithm.';
        }
        
        return `
            <h3>${this.name} Optimizer</h3>
            <p>${description}</p>
            <h4>Parameters:</h4>
            <ul>
                <li><strong>learning_rate:</strong> Step size for parameter updates</li>
            </ul>
            <h4>Inputs:</h4>
            <ul>
                <li><strong>model:</strong> Model to optimize</li>
                <li><strong>loss:</strong> Loss function to minimize</li>
            </ul>
            <h4>Outputs:</h4>
            <ul>
                <li><strong>optimized_model:</strong> Model configured with the optimizer</li>
            </ul>
        `;
    }
}

// Loss Components
class LossComponent extends MLComponent {
    constructor(config) {
        super({
            ...config,
            inputs: ['y_true', 'y_pred'],
            outputs: ['loss']
        });
    }

    async execute(inputs) {
        this.state = 'running';
        
        try {
            // In a real implementation, this would compute a loss value
            await new Promise(resolve => setTimeout(resolve, 150));
            
            this.state = 'completed';
            return {
                loss: {
                    type: this.id,
                    value: Math.random() // Placeholder for actual loss value
                }
            };
        } catch (error) {
            this.state = 'error';
            throw error;
        }
    }

    getDescription() {
        return `${this.name} loss function`;
    }

    getDocumentation() {
        let description = '';
        
        switch (this.id) {
            case 'categorical_crossentropy':
                description = 'Categorical crossentropy loss, used for multi-class classification.';
                break;
            case 'binary_crossentropy':
                description = 'Binary crossentropy loss, used for binary classification.';
                break;
            case 'mse':
                description = 'Mean squared error loss, used for regression.';
                break;
            default:
                description = 'A loss function.';
        }
        
        return `
            <h3>${this.name}</h3>
            <p>${description}</p>
            <h4>Inputs:</h4>
            <ul>
                <li><strong>y_true:</strong> Ground truth values</li>
                <li><strong>y_pred:</strong> Predicted values</li>
            </ul>
            <h4>Outputs:</h4>
            <ul>
                <li><strong>loss:</strong> Computed loss value</li>
            </ul>
        `;
    }
}

// Data Components
class DataLoaderComponent extends MLComponent {
    constructor(config) {
        super({
            ...config,
            inputs: [],
            outputs: ['data']
        });
    }

    validateParams() {
        if (this.id === 'csv_loader') {
            return typeof this.params.filepath === 'string' && this.params.filepath.length > 0;
        } else if (this.id === 'image_loader') {
            return typeof this.params.directory === 'string' && this.params.directory.length > 0;
        }
        
        return true;
    }

    async execute() {
        this.state = 'running';
        
        try {
            // In a real implementation, this would load data from a file or directory
            await new Promise(resolve => setTimeout(resolve, 800));
            
            this.state = 'completed';
            
            let data;
            
            if (this.id === 'csv_loader') {
                data = {
                    type: 'tabular',
                    source: this.params.filepath,
                    rows: 1000, // Placeholder
                    columns: 10, // Placeholder
                    sample: [] // Placeholder for actual data sample
                };
            } else if (this.id === 'image_loader') {
                data = {
                    type: 'image',
                    source: this.params.directory,
                    count: 500, // Placeholder
                    dimensions: this.params.target_size,
                    sample: [] // Placeholder for actual data sample
                };
            }
            
            return { data };
        } catch (error) {
            this.state = 'error';
            throw error;
        }
    }

    getDescription() {
        if (this.id === 'csv_loader') {
            return `CSV Loader (${this.params.filepath})`;
        } else if (this.id === 'image_loader') {
            return `Image Loader (${this.params.directory})`;
        }
        
        return `${this.name} data loader`;
    }

    getDocumentation() {
        if (this.id === 'csv_loader') {
            return `
                <h3>CSV Loader</h3>
                <p>Loads data from a CSV file.</p>
                <h4>Parameters:</h4>
                <ul>
                    <li><strong>filepath:</strong> Path to the CSV file</li>
                    <li><strong>delimiter:</strong> Column delimiter character</li>
                </ul>
                <h4>Outputs:</h4>
                <ul>
                    <li><strong>data:</strong> Loaded data</li>
                </ul>
            `;
        } else if (this.id === 'image_loader') {
            return `
                <h3>Image Loader</h3>
                <p>Loads images from a directory.</p>
                <h4>Parameters:</h4>
                <ul>
                    <li><strong>directory:</strong> Path to the image directory</li>
                    <li><strong>target_size:</strong> Target image dimensions [width, height]</li>
                </ul>
                <h4>Outputs:</h4>
                <ul>
                    <li><strong>data:</strong> Loaded images</li>
                </ul>
            `;
        }
        
        return super.getDocumentation();
    }
}

// Component Factory
class ComponentFactory {
    /**
     * Create a component instance
     * @param {string} type - Component type
     * @param {string} category - Component category
     * @param {string} instanceId - Instance ID
     * @param {Object} params - Component parameters
     * @param {Object} position - Component position
     * @returns {MLComponent} Component instance
     */
    static createComponent(type, category, instanceId, params, position) {
        const config = {
            id: type,
            instanceId,
            name: this.getComponentName(type, category),
            category,
            params,
            position
        };
        
        switch (category) {
            case 'layer':
                switch (type) {
                    case 'dense':
                        return new DenseLayer(config);
                    case 'conv2d':
                        return new Conv2DLayer(config);
                    default:
                        return new MLComponent(config);
                }
            
            case 'activation':
                return new ActivationComponent(config);
            
            case 'optimizer':
                return new OptimizerComponent(config);
            
            case 'loss':
                return new LossComponent(config);
            
            case 'data':
                return new DataLoaderComponent(config);
            
            default:
                return new MLComponent(config);
        }
    }
    
    /**
     * Get component name
     * @param {string} type - Component type
     * @param {string} category - Component category
     * @returns {string} Component name
     */
    static getComponentName(type, category) {
        const names = {
            // Layers
            'dense': 'Dense',
            'conv2d': 'Conv2D',
            'maxpool2d': 'MaxPooling2D',
            'dropout': 'Dropout',
            'flatten': 'Flatten',
            
            // Activations
            'relu': 'ReLU',
            'sigmoid': 'Sigmoid',
            'tanh': 'Tanh',
            'softmax': 'Softmax',
            
            // Optimizers
            'adam': 'Adam',
            'sgd': 'SGD',
            'rmsprop': 'RMSprop',
            
            // Losses
            'categorical_crossentropy': 'Categorical Crossentropy',
            'binary_crossentropy': 'Binary Crossentropy',
            'mse': 'Mean Squared Error',
            
            // Data
            'csv_loader': 'CSV Loader',
            'image_loader': 'Image Loader',
            'data_splitter': 'Train-Test Split',
            
            // Preprocessing
            'normalizer': 'Normalizer',
            'standard_scaler': 'Standard Scaler',
            'pca': 'PCA',
            
            // Evaluation
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1 Score',
            
            // Visualization
            'line_chart': 'Line Chart',
            'bar_chart': 'Bar Chart',
            'scatter_plot': 'Scatter Plot',
            'confusion_matrix': 'Confusion Matrix'
        };
        
        return names[type] || `Unknown ${category}`;
    }
}
