/**
 * ML Sandbox - Visualization JavaScript
 * Handles data visualization for ML components and workflows
 */

class Visualization {
    /**
     * Initialize visualization
     * @param {Object} options - Visualization options
     */
    constructor(options = {}) {
        this.options = {
            chartColors: {
                accuracy: '#28a745',
                loss: '#dc3545',
                precision: '#17a2b8',
                recall: '#ffc107',
                f1: '#6f42c1'
            },
            ...options
        };
        
        // Charts
        this.charts = {};
    }
    
    /**
     * Initialize training charts
     * @param {string} accuracyCanvasId - ID of accuracy chart canvas
     * @param {string} lossCanvasId - ID of loss chart canvas
     */
    initTrainingCharts(accuracyCanvasId = 'accuracy-chart', lossCanvasId = 'loss-chart') {
        const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
        const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        const textColor = isDarkMode ? '#e9ecef' : '#333';

        const chartConfig = {
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: {
                            color: gridColor
                        },
                        ticks: {
                            color: textColor
                        }
                    },
                    y: {
                        grid: {
                            color: gridColor
                        },
                        ticks: {
                            color: textColor
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: textColor
                        }
                    }
                }
            }
        };

        const accuracyCtx = document.getElementById(accuracyCanvasId).getContext('2d');
        const lossCtx = document.getElementById(lossCanvasId).getContext('2d');
        
        // Destroy existing charts if they exist
        if (this.charts.accuracy) {
            this.charts.accuracy.destroy();
        }
        
        if (this.charts.loss) {
            this.charts.loss.destroy();
        }
        
        // Create accuracy chart
        this.charts.accuracy = new Chart(accuracyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Accuracy',
                    data: [],
                    borderColor: this.options.chartColors.accuracy,
                    backgroundColor: this.hexToRgba(this.options.chartColors.accuracy, 0.1),
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                ...chartConfig.options,
                plugins: {
                    ...chartConfig.options.plugins,
                    title: {
                        display: true,
                        text: 'Accuracy',
                        color: textColor
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Accuracy',
                            color: textColor
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch',
                            color: textColor
                        }
                    }
                }
            }
        });
        
        // Create loss chart
        this.charts.loss = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Loss',
                    data: [],
                    borderColor: this.options.chartColors.loss,
                    backgroundColor: this.hexToRgba(this.options.chartColors.loss, 0.1),
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                ...chartConfig.options,
                plugins: {
                    ...chartConfig.options.plugins,
                    title: {
                        display: true,
                        text: 'Loss',
                        color: textColor
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Loss',
                            color: textColor
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch',
                            color: textColor
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Update training charts with new data
     * @param {Object} data - Training data
     */
    updateTrainingCharts(data) {
        if (!this.charts.accuracy || !this.charts.loss) {
            this.initTrainingCharts();
        }
        
        // Update accuracy chart
        if (data.accuracy !== undefined) {
            this.charts.accuracy.data.labels.push(data.epoch || this.charts.accuracy.data.labels.length + 1);
            this.charts.accuracy.data.datasets[0].data.push(data.accuracy);
            this.charts.accuracy.update();
        }
        
        // Update loss chart
        if (data.loss !== undefined) {
            this.charts.loss.data.labels.push(data.epoch || this.charts.loss.data.labels.length + 1);
            this.charts.loss.data.datasets[0].data.push(data.loss);
            this.charts.loss.update();
        }
    }
    
    /**
     * Create a confusion matrix visualization
     * @param {string} canvasId - ID of canvas element
     * @param {Array} matrix - Confusion matrix data
     * @param {Array} labels - Class labels
     */
    createConfusionMatrix(canvasId, matrix, labels) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }
        
        // Prepare data
        const data = {
            labels: labels,
            datasets: [{
                label: 'Confusion Matrix',
                data: this.flattenMatrix(matrix, labels),
                backgroundColor: this.getHeatmapColors(matrix),
                borderColor: '#ffffff',
                borderWidth: 1
            }]
        };
        
        // Create chart
        this.charts[canvasId] = new Chart(ctx, {
            type: 'matrix',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Confusion Matrix'
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const value = context.dataset.data[context.dataIndex];
                                return `Predicted: ${labels[value.x]}, Actual: ${labels[value.y]}, Count: ${value.v}`;
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Predicted'
                        },
                        ticks: {
                            stepSize: 1
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Actual'
                        },
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Flatten a matrix for Chart.js matrix chart
     * @param {Array} matrix - 2D matrix
     * @param {Array} labels - Labels
     * @returns {Array} Flattened matrix
     */
    flattenMatrix(matrix, labels) {
        const result = [];
        
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                result.push({
                    x: j,
                    y: i,
                    v: matrix[i][j]
                });
            }
        }
        
        return result;
    }
    
    /**
     * Get heatmap colors for confusion matrix
     * @param {Array} matrix - Confusion matrix
     * @returns {Array} Colors
     */
    getHeatmapColors(matrix) {
        const colors = [];
        let max = 0;
        
        // Find maximum value
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                if (matrix[i][j] > max) {
                    max = matrix[i][j];
                }
            }
        }
        
        // Generate colors
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                const intensity = matrix[i][j] / max;
                colors.push(this.getHeatmapColor(intensity));
            }
        }
        
        return colors;
    }
    
    /**
     * Get heatmap color based on intensity
     * @param {number} intensity - Color intensity (0-1)
     * @returns {string} Color
     */
    getHeatmapColor(intensity) {
        // Blue to red gradient
        const r = Math.floor(intensity * 255);
        const g = Math.floor((1 - intensity) * 100);
        const b = Math.floor((1 - intensity) * 255);
        
        return `rgb(${r}, ${g}, ${b})`;
    }
    
    /**
     * Create a bar chart
     * @param {string} canvasId - ID of canvas element
     * @param {Array} labels - X-axis labels
     * @param {Array} data - Data values
     * @param {string} title - Chart title
     */
    createBarChart(canvasId, labels, data, title = 'Bar Chart') {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }
        
        // Create chart
        this.charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: title,
                    data: data,
                    backgroundColor: this.hexToRgba(this.options.chartColors.precision, 0.7),
                    borderColor: this.options.chartColors.precision,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: title
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
     * Create a line chart
     * @param {string} canvasId - ID of canvas element
     * @param {Array} labels - X-axis labels
     * @param {Array} data - Data values
     * @param {string} title - Chart title
     */
    createLineChart(canvasId, labels, data, title = 'Line Chart') {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }
        
        // Create chart
        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: title,
                    data: data,
                    borderColor: this.options.chartColors.f1,
                    backgroundColor: this.hexToRgba(this.options.chartColors.f1, 0.1),
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: title
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
     * Create a scatter plot
     * @param {string} canvasId - ID of canvas element
     * @param {Array} data - Data points [{x, y}]
     * @param {string} title - Chart title
     */
    createScatterPlot(canvasId, data, title = 'Scatter Plot') {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }
        
        // Create chart
        this.charts[canvasId] = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: title,
                    data: data,
                    backgroundColor: this.options.chartColors.recall
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: title
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom'
                    },
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    /**
     * Create a workflow visualization
     * @param {string} containerId - ID of container element
     * @param {Object} workflow - Workflow data
     */
    createWorkflowVisualization(containerId, workflow) {
        const container = document.getElementById(containerId);
        
        // Clear container
        container.innerHTML = '';
        
        // Create SVG element
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        container.appendChild(svg);
        
        // Create nodes
        workflow.nodes.forEach(node => {
            const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            nodeGroup.setAttribute('transform', `translate(${node.position.x}, ${node.position.y})`);
            
            // Create node rectangle
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('width', '150');
            rect.setAttribute('height', '80');
            rect.setAttribute('rx', '5');
            rect.setAttribute('ry', '5');
            rect.setAttribute('fill', this.getCategoryColor(node.category));
            nodeGroup.appendChild(rect);
            
            // Create node title
            const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            title.setAttribute('x', '75');
            title.setAttribute('y', '25');
            title.setAttribute('text-anchor', 'middle');
            title.setAttribute('fill', 'white');
            title.textContent = node.name;
            nodeGroup.appendChild(title);
            
            // Create node type
            const type = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            type.setAttribute('x', '75');
            type.setAttribute('y', '45');
            type.setAttribute('text-anchor', 'middle');
            type.setAttribute('fill', 'white');
            type.setAttribute('font-size', '12');
            type.textContent = node.category;
            nodeGroup.appendChild(type);
            
            svg.appendChild(nodeGroup);
        });
        
        // Create connections
        workflow.connections.forEach(conn => {
            const sourceNode = workflow.nodes.find(node => node.id === conn.source);
            const targetNode = workflow.nodes.find(node => node.id === conn.target);
            
            if (sourceNode && targetNode) {
                const sourceX = sourceNode.position.x + 150; // Right side of source node
                const sourceY = sourceNode.position.y + 40; // Middle of source node
                const targetX = targetNode.position.x; // Left side of target node
                const targetY = targetNode.position.y + 40; // Middle of target node
                
                // Create path
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                const controlPointX = (sourceX + targetX) / 2;
                path.setAttribute('d', `M ${sourceX} ${sourceY} C ${controlPointX} ${sourceY}, ${controlPointX} ${targetY}, ${targetX} ${targetY}`);
                path.setAttribute('stroke', '#4a6bff');
                path.setAttribute('stroke-width', '2');
                path.setAttribute('fill', 'none');
                
                // Create arrow
                const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
                arrow.setAttribute('points', `${targetX},${targetY} ${targetX-10},${targetY-5} ${targetX-10},${targetY+5}`);
                arrow.setAttribute('fill', '#4a6bff');
                
                svg.appendChild(path);
                svg.appendChild(arrow);
            }
        });
    }
    
    /**
     * Get color for component category
     * @param {string} category - Component category
     * @returns {string} Color
     */
    getCategoryColor(category) {
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
        
        return colors[category] || '#6c757d';
    }
    
    /**
     * Convert hex color to rgba
     * @param {string} hex - Hex color
     * @param {number} alpha - Alpha value
     * @returns {string} RGBA color
     */
    hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
}

// Export the Visualization class
window.Visualization = Visualization;
