class HelpSystem {
    constructor() {
        this.helpTopics = {
            general: {
                title: 'Getting Started',
                sections: [
                    {
                        title: 'Overview',
                        content: 'ML Sandbox is an interactive AI/ML development environment that allows you to build and test machine learning experiments using a drag-and-drop interface.'
                    },
                    {
                        title: 'Basic Navigation',
                        content: 'Use the top navigation bar to access core functions like creating new projects, saving work, and accessing settings. The left sidebar contains ML components you can drag onto the canvas.'
                    }
                ]
            },
            components: {
                title: 'ML Components',
                sections: [
                    {
                        title: 'Neural Network Layers',
                        content: 'Drag and drop layers like Dense, Conv2D, and LSTM to build your neural network architecture.'
                    },
                    {
                        title: 'Data Processing',
                        content: 'Use data processing blocks to normalize, transform, and prepare your data for training.'
                    },
                    {
                        title: 'Training',
                        content: 'Configure training parameters, optimizers, and loss functions using the training components.'
                    }
                ]
            },
            workflows: {
                title: 'Working with Workflows',
                sections: [
                    {
                        title: 'Creating Workflows',
                        content: 'Start by dragging components onto the canvas. Connect them by clicking and dragging between component ports.'
                    },
                    {
                        title: 'Saving & Loading',
                        content: 'Save your workflows using the Save button. Load existing workflows using the Load button or from the recent workflows list.'
                    }
                ]
            },
            troubleshooting: {
                title: 'Troubleshooting',
                sections: [
                    {
                        title: 'Common Issues',
                        content: 'Find solutions to common problems and error messages.'
                    },
                    {
                        title: 'Support',
                        content: 'Contact support or visit our documentation for additional help.'
                    }
                ]
            }
        };

        this.initializeTooltips();
    }

    showHelpModal(topic = null) {
        const modal = document.getElementById('help-modal');
        const content = document.getElementById('help-content');
        const search = document.getElementById('help-search');

        if (topic && this.helpTopics[topic]) {
            this.displayTopic(topic);
        } else {
            this.displayTopicsList();
        }

        modal.style.display = 'block';
    }

    displayTopic(topicId) {
        const topic = this.helpTopics[topicId];
        const content = document.getElementById('help-content');
        
        content.innerHTML = `
            <h2>${topic.title}</h2>
            ${topic.sections.map(section => `
                <div class="help-section">
                    <h3>${section.title}</h3>
                    <p>${section.content}</p>
                </div>
            `).join('')}
        `;
    }

    displayTopicsList() {
        const content = document.getElementById('help-content');
        
        content.innerHTML = `
            <h2>Help Topics</h2>
            <div class="help-topics-grid">
                ${Object.entries(this.helpTopics).map(([id, topic]) => `
                    <div class="help-topic-card" onclick="helpSystem.displayTopic('${id}')">
                        <h3>${topic.title}</h3>
                        <p>${topic.sections[0].content.substring(0, 100)}...</p>
                    </div>
                `).join('')}
            </div>
        `;
    }

    searchHelp(query) {
        query = query.toLowerCase();
        const results = [];

        for (const [topicId, topic] of Object.entries(this.helpTopics)) {
            const topicResults = topic.sections.filter(section =>
                section.title.toLowerCase().includes(query) ||
                section.content.toLowerCase().includes(query)
            );

            if (topicResults.length > 0) {
                results.push({
                    topicId,
                    topicTitle: topic.title,
                    sections: topicResults
                });
            }
        }

        this.displaySearchResults(results);
    }

    displaySearchResults(results) {
        const content = document.getElementById('help-content');
        
        if (results.length === 0) {
            content.innerHTML = '<p>No results found</p>';
            return;
        }

        content.innerHTML = `
            <h2>Search Results</h2>
            ${results.map(result => `
                <div class="help-search-result">
                    <h3>${result.topicTitle}</h3>
                    ${result.sections.map(section => `
                        <div class="help-section">
                            <h4>${section.title}</h4>
                            <p>${section.content}</p>
                        </div>
                    `).join('')}
                </div>
            `).join('')}
        `;
    }

    initializeTooltips() {
        const tooltips = {
            'new-project': 'Create a new ML project',
            'save-project': 'Save current project',
            'load-project': 'Load existing project',
            'export-project': 'Export project in various formats',
            'settings': 'Configure application settings',
            'run-workflow': 'Execute current workflow',
            'stop-workflow': 'Stop workflow execution'
        };

        for (const [id, text] of Object.entries(tooltips)) {
            const element = document.getElementById(id);
            if (element) {
                element.setAttribute('data-tooltip', text);
            }
        }
    }
}

// Initialize help system
const helpSystem = new HelpSystem();