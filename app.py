import os
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO

# Import our custom modules
from workflow_manager import WorkflowManager
from error_handler import ErrorHandler
from data_manager import DataManager
from model_registry import ModelRegistry
from ml_executor import MLExecutor
from tf_executor import TFExecutor
from torch_executor import TorchExecutor

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_development_only')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize our modules
error_handler = ErrorHandler()
workflow_manager = WorkflowManager("workflows")
data_manager = DataManager("datasets")
model_registry = ModelRegistry("models")

# Initialize ML executor
executor = MLExecutor()

# Initialize ML executor with appropriate framework executor
framework = os.environ.get('ML_FRAMEWORK', 'tensorflow').lower()
if framework == 'tensorflow':
    executor.framework_executor = TFExecutor()
elif framework == 'pytorch':
    executor.framework_executor = TorchExecutor()
else:
    raise ValueError(f"Unsupported ML framework: {framework}")

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Render the main application page."""
    return render_template('index.html')

@app.route('/api/components', methods=['GET'])
def get_components():
    """Return a list of available ML components."""
    components = {
        'layers': [
            {'id': 'dense', 'name': 'Dense', 'category': 'layer', 'params': {'units': 64, 'activation': 'relu'}},
            {'id': 'conv2d', 'name': 'Conv2D', 'category': 'layer', 'params': {'filters': 32, 'kernel_size': 3, 'activation': 'relu'}},
            {'id': 'maxpool2d', 'name': 'MaxPooling2D', 'category': 'layer', 'params': {'pool_size': 2}},
            {'id': 'dropout', 'name': 'Dropout', 'category': 'layer', 'params': {'rate': 0.25}},
            {'id': 'flatten', 'name': 'Flatten', 'category': 'layer', 'params': {}}
        ],
        'activations': [
            {'id': 'relu', 'name': 'ReLU', 'category': 'activation'},
            {'id': 'sigmoid', 'name': 'Sigmoid', 'category': 'activation'},
            {'id': 'tanh', 'name': 'Tanh', 'category': 'activation'},
            {'id': 'softmax', 'name': 'Softmax', 'category': 'activation'}
        ],
        'optimizers': [
            {'id': 'adam', 'name': 'Adam', 'category': 'optimizer', 'params': {'learning_rate': 0.001}},
            {'id': 'sgd', 'name': 'SGD', 'category': 'optimizer', 'params': {'learning_rate': 0.01}},
            {'id': 'rmsprop', 'name': 'RMSprop', 'category': 'optimizer', 'params': {'learning_rate': 0.001}}
        ],
        'losses': [
            {'id': 'categorical_crossentropy', 'name': 'Categorical Crossentropy', 'category': 'loss'},
            {'id': 'binary_crossentropy', 'name': 'Binary Crossentropy', 'category': 'loss'},
            {'id': 'mse', 'name': 'Mean Squared Error', 'category': 'loss'}
        ],
        'data': [
            {'id': 'csv_loader', 'name': 'CSV Loader', 'category': 'data', 'params': {'filepath': '', 'delimiter': ','}},
            {'id': 'image_loader', 'name': 'Image Loader', 'category': 'data', 'params': {'directory': '', 'target_size': [224, 224]}},
            {'id': 'data_splitter', 'name': 'Train-Test Split', 'category': 'data', 'params': {'test_size': 0.2, 'random_state': 42}}
        ],
        'preprocessing': [
            {'id': 'normalizer', 'name': 'Normalizer', 'category': 'preprocessing', 'params': {}},
            {'id': 'standard_scaler', 'name': 'Standard Scaler', 'category': 'preprocessing', 'params': {}},
            {'id': 'pca', 'name': 'PCA', 'category': 'preprocessing', 'params': {'n_components': 2}}
        ],
        'evaluation': [
            {'id': 'accuracy', 'name': 'Accuracy', 'category': 'evaluation'},
            {'id': 'precision', 'name': 'Precision', 'category': 'evaluation'},
            {'id': 'recall', 'name': 'Recall', 'category': 'evaluation'},
            {'id': 'f1', 'name': 'F1 Score', 'category': 'evaluation'}
        ],
        'visualization': [
            {'id': 'line_chart', 'name': 'Line Chart', 'category': 'visualization'},
            {'id': 'bar_chart', 'name': 'Bar Chart', 'category': 'visualization'},
            {'id': 'scatter_plot', 'name': 'Scatter Plot', 'category': 'visualization'},
            {'id': 'confusion_matrix', 'name': 'Confusion Matrix', 'category': 'visualization'}
        ]
    }
    return jsonify(components)

@app.route('/api/execute', methods=['POST'])
def execute_workflow():
    """Execute the ML workflow defined by the user."""
    try:
        workflow_data = request.json
        
        # Validate workflow
        validation_result = error_handler.validate_workflow(workflow_data)
        if not validation_result["valid"]:
            return jsonify({
                'status': 'error',
                'message': 'Invalid workflow',
                'errors': validation_result["issues"]
            }), 400
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        # Monitor execution start
        error_handler.monitor_execution(execution_id, "started")
        
        # Emit execution started event
        socketio.emit('execution_update', {
            'status': 'started',
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat()
        })
        
        # Execute workflow
        try:
            result = executor.execute_workflow(workflow_data)
            
            # If execution successful, save workflow
            if 'workflow_id' in workflow_data:
                # Update existing workflow
                workflow_manager.update_workflow({
                    'id': workflow_data['workflow_id'],
                    'updated_at': datetime.now().isoformat(),
                    'last_execution': {
                        'execution_id': execution_id,
                        'status': 'success',
                        'timestamp': datetime.now().isoformat(),
                        'metrics': result.get('metrics', {})
                    }
                })
            else:
                # Create new workflow
                workflow_id = workflow_manager.create_workflow(
                    name=workflow_data.get('name', 'Untitled Workflow'),
                    description=workflow_data.get('description', '')
                )['id']
                
                # Update workflow with nodes and connections
                workflow_manager.update_workflow({
                    'id': workflow_id,
                    'nodes': workflow_data.get('nodes', {}),
                    'connections': workflow_data.get('connections', []),
                    'last_execution': {
                        'execution_id': execution_id,
                        'status': 'success',
                        'timestamp': datetime.now().isoformat(),
                        'metrics': result.get('metrics', {})
                    }
                })
                
                # Add workflow_id to result
                result['workflow_id'] = workflow_id
            
            # Monitor execution completion
            error_handler.monitor_execution(execution_id, "completed", result.get('stats', {}))
            
            # Emit execution completed event
            socketio.emit('execution_update', {
                'status': 'completed',
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat(),
                'result': result
            })
            
            return jsonify({
                'status': 'success',
                'execution_id': execution_id,
                'result': result
            })
            
        except Exception as e:
            # Handle execution error
            error_info = error_handler.handle_exception(
                e, 
                component='workflow_execution',
                context={'workflow_data': workflow_data, 'execution_id': execution_id}
            )
            
            # Monitor execution error
            error_handler.monitor_execution(execution_id, "error", {'error': str(e)})
            
            # Emit execution error event
            socketio.emit('execution_update', {
                'status': 'error',
                'execution_id': execution_id,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            
            return jsonify({
                'status': 'error',
                'message': str(e),
                'execution_id': execution_id,
                'error_details': error_info
            }), 500
            
    except Exception as e:
        # Handle general error
        error_info = error_handler.handle_exception(
            e, 
            component='api_execute',
            context={'request_data': request.json}
        )
        
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

# Workflow Management Routes
@app.route('/api/workflows', methods=['GET'])
def list_workflows():
    """List all available workflows."""
    try:
        workflows = workflow_manager.list_workflows()
        return jsonify(workflows)
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='list_workflows')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/workflows/<workflow_id>', methods=['GET'])
def get_workflow(workflow_id):
    """Get a specific workflow by ID."""
    try:
        workflow = workflow_manager.load_workflow(workflow_id)
        return jsonify(workflow)
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='get_workflow')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/workflows', methods=['POST'])
def create_workflow():
    """Create a new workflow."""
    try:
        data = request.json
        workflow = workflow_manager.create_workflow(
            name=data.get('name', 'Untitled Workflow'),
            description=data.get('description', '')
        )
        return jsonify(workflow)
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='create_workflow')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/workflows/<workflow_id>', methods=['PUT'])
def update_workflow(workflow_id):
    """Update an existing workflow."""
    try:
        data = request.json
        workflow = workflow_manager.load_workflow(workflow_id)
        updated_workflow = workflow_manager.update_workflow(data)
        return jsonify(updated_workflow)
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='update_workflow')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/workflows/<workflow_id>/version', methods=['POST'])
def create_workflow_version(workflow_id):
    """Create a new version of a workflow."""
    try:
        workflow = workflow_manager.load_workflow(workflow_id)
        new_version = workflow_manager.create_version()
        return jsonify(new_version)
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='create_workflow_version')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

# Dataset Management Routes
@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """List all available datasets."""
    try:
        datasets = data_manager.list_datasets()
        return jsonify(datasets)
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='list_datasets')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """Get information about a specific dataset."""
    try:
        dataset_info = data_manager.get_dataset_info(dataset_id)
        return jsonify(dataset_info)
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='get_dataset')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset():
    """Upload a new dataset."""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file part in the request'
            }), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
            
        # Save file to uploads directory
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Load dataset
        dataset_name = request.form.get('name', None)
        dataset_info = data_manager.load_dataset(filename, dataset_name)
        
        return jsonify({
            'status': 'success',
            'message': 'Dataset uploaded successfully',
            'dataset': dataset_info
        })
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='upload_dataset')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/datasets/<dataset_id>/preprocess', methods=['POST'])
def preprocess_dataset(dataset_id):
    """Preprocess a dataset."""
    try:
        operations = request.json.get('operations', [])
        result = data_manager.preprocess_dataset(dataset_id, operations)
        return jsonify({
            'status': 'success',
            'message': 'Dataset preprocessed successfully',
            'result': result
        })
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='preprocess_dataset')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/datasets/<dataset_id>/split', methods=['POST'])
def split_dataset(dataset_id):
    """Split a dataset into train, validation, and test sets."""
    try:
        data = request.json
        result = data_manager.split_dataset(
            dataset_id=dataset_id,
            target_column=data.get('target_column'),
            test_size=data.get('test_size', 0.2),
            val_size=data.get('val_size', 0.1),
            random_state=data.get('random_state', 42),
            stratify=data.get('stratify', False)
        )
        return jsonify({
            'status': 'success',
            'message': 'Dataset split successfully',
            'result': result
        })
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='split_dataset')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

# Model Registry Routes
@app.route('/api/models', methods=['GET'])
def list_models():
    """List all registered models."""
    try:
        framework = request.args.get('framework')
        tags = request.args.getlist('tags')
        models = model_registry.list_models(framework, tags if tags else None)
        return jsonify(models)
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='list_models')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get information about a specific model."""
    try:
        model_info = model_registry.get_model_info(model_id)
        return jsonify(model_info)
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='get_model')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/models/<model_id>', methods=['PUT'])
def update_model(model_id):
    """Update model information."""
    try:
        updates = request.json
        model_info = model_registry.update_model_info(model_id, updates)
        return jsonify(model_info)
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='update_model')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a model from the registry."""
    try:
        result = model_registry.delete_model(model_id)
        return jsonify({
            'status': 'success',
            'message': f'Model {model_id} deleted successfully'
        })
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='delete_model')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/models/compare', methods=['POST'])
def compare_models():
    """Compare multiple models."""
    try:
        model_ids = request.json.get('model_ids', [])
        if not model_ids:
            return jsonify({
                'status': 'error',
                'message': 'No model IDs provided for comparison'
            }), 400
            
        comparison = model_registry.compare_models(model_ids)
        return jsonify({
            'status': 'success',
            'comparison': comparison
        })
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='compare_models')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/models/import', methods=['POST'])
def import_models():
    """Import models into the registry."""
    try:
        models_data = request.json.get('models', [])
        imported_models = []
        
        for model_data in models_data:
            model_info = model_registry.import_model(
                model_path=model_data.get('model_path'),
                framework=model_data.get('framework'),
                name=model_data.get('name'),
                description=model_data.get('description'),
                tags=model_data.get('tags')
            )
            imported_models.append(model_info)
            
        return jsonify({
            'status': 'success',
            'message': f'Successfully imported {len(imported_models)} models',
            'models': imported_models
        })
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='import_models')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/datasets/import', methods=['POST'])
def import_datasets():
    """Import datasets into the data manager."""
    try:
        datasets_data = request.json.get('datasets', [])
        imported_datasets = []
        
        for dataset_data in datasets_data:
            dataset_info = data_manager.import_dataset(
                file_path=dataset_data.get('file_path'),
                dataset_name=dataset_data.get('name'),
                metadata=dataset_data.get('metadata')
            )
            imported_datasets.append(dataset_info)
            
        return jsonify({
            'status': 'success',
            'message': f'Successfully imported {len(imported_datasets)} datasets',
            'datasets': imported_datasets
        })
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='import_datasets')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

# Error Handling Routes
@app.route('/api/errors', methods=['GET'])
def get_errors():
    """Get recent errors."""
    try:
        limit = request.args.get('limit', type=int)
        component = request.args.get('component')
        errors = error_handler.get_errors(limit, component)
        return jsonify(errors)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/warnings', methods=['GET'])
def get_warnings():
    """Get recent warnings."""
    try:
        limit = request.args.get('limit', type=int)
        component = request.args.get('component')
        warnings = error_handler.get_warnings(limit, component)
        return jsonify(warnings)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection to WebSocket."""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection from WebSocket."""
    print('Client disconnected')

@socketio.on('execution_progress')
def handle_execution_progress(data):
    """Handle execution progress updates."""
    # Broadcast progress to all clients
    socketio.emit('execution_update', {
        'status': 'progress',
        'execution_id': data.get('execution_id'),
        'progress': data.get('progress'),
        'current_step': data.get('current_step'),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/models/<model_id>/export', methods=['POST'])
def export_model(model_id):
    """Export a model in the specified format."""
    try:
        export_format = request.args.get('format', 'savedmodel')
        export_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        export_path = model_registry.export_model(
            model_id=model_id,
            export_dir=export_dir,
            format=export_format
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Model exported successfully',
            'exportPath': export_path
        })
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='export_model')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/datasets/<dataset_id>/export', methods=['POST'])
def export_dataset(dataset_id):
    """Export a dataset in the specified format."""
    try:
        export_format = request.args.get('format', 'csv')
        export_path = data_manager.export_dataset(
            dataset_id=dataset_id,
            format=export_format
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Dataset exported successfully',
            'exportPath': export_path
        })
    except Exception as e:
        error_info = error_handler.handle_exception(e, component='export_dataset')
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_details': error_info
        }), 500

@app.route('/api/models/download/<path:filename>')
def download_model(filename):
    """Download an exported model file."""
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], 'exports'),
        filename,
        as_attachment=True
    )

@app.route('/api/datasets/download/<path:filename>')
def download_dataset(filename):
    """Download an exported dataset file."""
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], 'exports'),
        filename,
        as_attachment=True
    )

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')
