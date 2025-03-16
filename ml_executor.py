"""
ML Executor - Handles the execution of ML workflows
"""
import os
import time
import json
import traceback
import numpy as np
import tensorflow as tf
import torch
import pandas as pd
from PIL import Image
from io import BytesIO
import psutil

class MLExecutor:
    """
    Executes ML workflows based on component configurations
    """
    def __init__(self):
        self.components = {}
        self.execution_stats = {
            'start_time': None,
            'end_time': None,
            'memory_usage': 0,
            'errors': []
        }
        self.execution_status = 'idle'
        self.execution_results = {}
        self.framework = 'tensorflow'  # Default framework
        
    def register_component(self, component_id, component_data):
        """
        Register a component for execution
        
        Args:
            component_id (str): Unique ID for the component
            component_data (dict): Component configuration data
        """
        self.components[component_id] = component_data
        
    def unregister_component(self, component_id):
        """
        Unregister a component
        
        Args:
            component_id (str): ID of the component to unregister
        """
        if component_id in self.components:
            del self.components[component_id]
            
    def clear_components(self):
        """Clear all registered components"""
        self.components = {}
        
    def set_framework(self, framework):
        """
        Set the ML framework to use
        
        Args:
            framework (str): 'tensorflow' or 'pytorch'
        """
        if framework in ['tensorflow', 'pytorch']:
            self.framework = framework
        else:
            raise ValueError(f"Unsupported framework: {framework}. Use 'tensorflow' or 'pytorch'")
            
    def execute_workflow(self, workflow, on_progress=None):
        """
        Execute an ML workflow
        
        Args:
            workflow (dict): Workflow configuration with nodes and connections
            on_progress (function): Callback function for progress updates
            
        Returns:
            dict: Execution results
        """
        try:
            self.execution_status = 'running'
            self.execution_stats['start_time'] = time.time()
            self.execution_stats['errors'] = []
            
            # Reset execution results
            self.execution_results = {}
            
            # Get execution order
            execution_order = self._determine_execution_order(workflow)
            
            # Execute components in order
            for component_id in execution_order:
                if self.execution_status == 'stopped':
                    break
                    
                component = workflow['nodes'].get(component_id)
                if not component:
                    raise ValueError(f"Component {component_id} not found in workflow")
                
                # Update progress
                if on_progress:
                    progress = {
                        'component_id': component_id,
                        'status': 'executing',
                        'message': f"Executing {component['name']}..."
                    }
                    on_progress(progress)
                
                # Execute component
                try:
                    result = self._execute_component(component, workflow)
                    self.execution_results[component_id] = result
                    
                    # Update progress
                    if on_progress:
                        progress = {
                            'component_id': component_id,
                            'status': 'completed',
                            'message': f"Completed {component['name']}"
                        }
                        on_progress(progress)
                        
                except Exception as e:
                    error_msg = f"Error executing {component['name']}: {str(e)}"
                    self.execution_stats['errors'].append({
                        'component_id': component_id,
                        'error': error_msg,
                        'traceback': traceback.format_exc()
                    })
                    
                    # Update progress
                    if on_progress:
                        progress = {
                            'component_id': component_id,
                            'status': 'error',
                            'message': error_msg
                        }
                        on_progress(progress)
                    
                    # Stop execution on error
                    break
                
                # Update memory usage
                process = psutil.Process(os.getpid())
                self.execution_stats['memory_usage'] = process.memory_info().rss / (1024 * 1024)  # MB
            
            self.execution_status = 'completed' if not self.execution_stats['errors'] else 'error'
            self.execution_stats['end_time'] = time.time()
            
            return {
                'status': self.execution_status,
                'results': self.execution_results,
                'stats': self._get_execution_stats()
            }
            
        except Exception as e:
            self.execution_status = 'error'
            self.execution_stats['end_time'] = time.time()
            self.execution_stats['errors'].append({
                'component_id': 'workflow',
                'error': f"Workflow execution error: {str(e)}",
                'traceback': traceback.format_exc()
            })
            
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'stats': self._get_execution_stats()
            }
    
    def stop_execution(self):
        """Stop the current execution"""
        self.execution_status = 'stopped'
        self.execution_stats['end_time'] = time.time()
        
    def _determine_execution_order(self, workflow):
        """
        Determine the execution order for components in a workflow
        
        Args:
            workflow (dict): Workflow configuration
            
        Returns:
            list: Component IDs in execution order
        """
        nodes = workflow['nodes']
        connections = workflow['connections']
        
        # Build dependency graph
        graph = {node_id: [] for node_id in nodes}
        for conn in connections:
            source = conn['source']
            target = conn['target']
            graph[target].append(source)
        
        # Topological sort
        visited = set()
        temp = set()
        order = []
        
        def visit(node_id):
            if node_id in temp:
                raise ValueError(f"Circular dependency detected involving {node_id}")
            if node_id in visited:
                return
            
            temp.add(node_id)
            for dependency in graph[node_id]:
                visit(dependency)
            
            temp.remove(node_id)
            visited.add(node_id)
            order.append(node_id)
        
        for node_id in graph:
            if node_id not in visited:
                visit(node_id)
                
        # Reverse to get correct execution order
        return order[::-1]
    
    def _execute_component(self, component, workflow):
        """
        Execute a single component
        
        Args:
            component (dict): Component configuration
            workflow (dict): Complete workflow for context
            
        Returns:
            object: Component execution result
        """
        component_type = component['type']
        category = component['category']
        params = component.get('params', {})
        
        # Get input data from connected components
        inputs = self._get_component_inputs(component['id'], workflow)
        
        # Execute based on component type and category
        if category == 'data':
            return self._execute_data_component(component_type, params, inputs)
        elif category == 'preprocessing':
            return self._execute_preprocessing_component(component_type, params, inputs)
        elif category == 'layer':
            return self._execute_layer_component(component_type, params, inputs)
        elif category == 'activation':
            return self._execute_activation_component(component_type, params, inputs)
        elif category == 'optimizer':
            return self._execute_optimizer_component(component_type, params, inputs)
        elif category == 'loss':
            return self._execute_loss_component(component_type, params, inputs)
        elif category == 'model':
            return self._execute_model_component(component_type, params, inputs)
        elif category == 'training':
            return self._execute_training_component(component_type, params, inputs)
        elif category == 'evaluation':
            return self._execute_evaluation_component(component_type, params, inputs)
        elif category == 'visualization':
            return self._execute_visualization_component(component_type, params, inputs)
        else:
            raise ValueError(f"Unsupported component category: {category}")
    
    def _get_component_inputs(self, component_id, workflow):
        """
        Get input data for a component from its connections
        
        Args:
            component_id (str): ID of the component
            workflow (dict): Workflow configuration
            
        Returns:
            dict: Input data for the component
        """
        inputs = {}
        
        for conn in workflow['connections']:
            if conn['target'] == component_id:
                source_id = conn['source']
                source_port = conn.get('sourcePort', 'output')
                target_port = conn.get('targetPort', 'input')
                
                if source_id in self.execution_results:
                    source_result = self.execution_results[source_id]
                    
                    # Handle different output formats
                    if isinstance(source_result, dict) and source_port in source_result:
                        inputs[target_port] = source_result[source_port]
                    else:
                        inputs[target_port] = source_result
        
        return inputs
    
    def _get_execution_stats(self):
        """
        Get execution statistics
        
        Returns:
            dict: Execution statistics
        """
        stats = {
            'duration': 0,
            'memory_usage': self.execution_stats['memory_usage'],
            'error_count': len(self.execution_stats['errors'])
        }
        
        if self.execution_stats['start_time'] and self.execution_stats['end_time']:
            stats['duration'] = self.execution_stats['end_time'] - self.execution_stats['start_time']
            
        return stats
    
    # Component execution methods
    def _execute_data_component(self, component_type, params, inputs):
        """Execute a data component"""
        if component_type == 'csv_loader':
            return self._execute_csv_loader(params)
        elif component_type == 'image_loader':
            return self._execute_image_loader(params)
        elif component_type == 'numpy_array':
            return self._execute_numpy_array(params)
        elif component_type == 'random_data':
            return self._execute_random_data(params)
        else:
            raise ValueError(f"Unsupported data component type: {component_type}")
    
    def _execute_preprocessing_component(self, component_type, params, inputs):
        """Execute a preprocessing component"""
        if component_type == 'normalization':
            return self._execute_normalization(params, inputs)
        elif component_type == 'one_hot_encoding':
            return self._execute_one_hot_encoding(params, inputs)
        elif component_type == 'train_test_split':
            return self._execute_train_test_split(params, inputs)
        elif component_type == 'image_resize':
            return self._execute_image_resize(params, inputs)
        else:
            raise ValueError(f"Unsupported preprocessing component type: {component_type}")
    
    def _execute_layer_component(self, component_type, params, inputs):
        """Execute a layer component"""
        if self.framework == 'tensorflow':
            return self._execute_tf_layer(component_type, params, inputs)
        else:
            return self._execute_torch_layer(component_type, params, inputs)
    
    def _execute_activation_component(self, component_type, params, inputs):
        """Execute an activation component"""
        if self.framework == 'tensorflow':
            return self._execute_tf_activation(component_type, params, inputs)
        else:
            return self._execute_torch_activation(component_type, params, inputs)
    
    def _execute_optimizer_component(self, component_type, params, inputs):
        """Execute an optimizer component"""
        if self.framework == 'tensorflow':
            return self._execute_tf_optimizer(component_type, params, inputs)
        else:
            return self._execute_torch_optimizer(component_type, params, inputs)
    
    def _execute_loss_component(self, component_type, params, inputs):
        """Execute a loss component"""
        if self.framework == 'tensorflow':
            return self._execute_tf_loss(component_type, params, inputs)
        else:
            return self._execute_torch_loss(component_type, params, inputs)
    
    def _execute_model_component(self, component_type, params, inputs):
        """Execute a model component"""
        if self.framework == 'tensorflow':
            return self._execute_tf_model(component_type, params, inputs)
        else:
            return self._execute_torch_model(component_type, params, inputs)
    
    def _execute_training_component(self, component_type, params, inputs):
        """Execute a training component"""
        if self.framework == 'tensorflow':
            return self._execute_tf_training(component_type, params, inputs)
        else:
            return self._execute_torch_training(component_type, params, inputs)
    
    def _execute_evaluation_component(self, component_type, params, inputs):
        """Execute an evaluation component"""
        if self.framework == 'tensorflow':
            return self._execute_tf_evaluation(component_type, params, inputs)
        else:
            return self._execute_torch_evaluation(component_type, params, inputs)
    
    def _execute_visualization_component(self, component_type, params, inputs):
        """Execute a visualization component"""
        if component_type == 'confusion_matrix':
            return self._execute_confusion_matrix(params, inputs)
        elif component_type == 'learning_curves':
            return self._execute_learning_curves(params, inputs)
        elif component_type == 'feature_importance':
            return self._execute_feature_importance(params, inputs)
        else:
            raise ValueError(f"Unsupported visualization component type: {component_type}")
    
    # Data component implementations
    def _execute_csv_loader(self, params):
        """Load data from a CSV file"""
        file_path = params.get('file_path')
        if not file_path:
            raise ValueError("CSV loader requires a file_path parameter")
        
        try:
            df = pd.read_csv(file_path)
            return {
                'data': df,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict()
            }
        except Exception as e:
            raise RuntimeError(f"Error loading CSV file: {str(e)}")
    
    def _execute_image_loader(self, params):
        """Load image data from a directory"""
        directory = params.get('directory')
        if not directory:
            raise ValueError("Image loader requires a directory parameter")
        
        try:
            images = []
            labels = []
            class_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
            
            for class_idx, class_dir in enumerate(class_dirs):
                class_path = os.path.join(directory, class_dir)
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        images.append(img_array)
                        labels.append(class_idx)
            
            return {
                'images': np.array(images),
                'labels': np.array(labels),
                'num_classes': len(class_dirs),
                'class_names': class_dirs,
                'shape': (len(images),) + images[0].shape
            }
        except Exception as e:
            raise RuntimeError(f"Error loading images: {str(e)}")
    
    def _execute_numpy_array(self, params):
        """Create a numpy array from parameters"""
        data = params.get('data')
        shape = params.get('shape')
        
        if data:
            try:
                return np.array(json.loads(data))
            except:
                raise ValueError("Invalid data format for numpy array")
        elif shape:
            try:
                shape_tuple = tuple(json.loads(shape))
                return np.zeros(shape_tuple)
            except:
                raise ValueError("Invalid shape format for numpy array")
        else:
            raise ValueError("Numpy array requires either data or shape parameter")
    
    def _execute_random_data(self, params):
        """Generate random data"""
        distribution = params.get('distribution', 'normal')
        shape = params.get('shape', '[100, 10]')
        
        try:
            shape_tuple = tuple(json.loads(shape))
            
            if distribution == 'normal':
                data = np.random.normal(0, 1, shape_tuple)
            elif distribution == 'uniform':
                data = np.random.uniform(0, 1, shape_tuple)
            elif distribution == 'bernoulli':
                data = np.random.binomial(1, 0.5, shape_tuple)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")
            
            return {
                'data': data,
                'shape': data.shape,
                'distribution': distribution
            }
        except Exception as e:
            raise RuntimeError(f"Error generating random data: {str(e)}")
    
    # Preprocessing component implementations
    def _execute_normalization(self, params, inputs):
        """Normalize input data"""
        if 'input' not in inputs:
            raise ValueError("Normalization component requires input data")
        
        method = params.get('method', 'min_max')
        data = inputs['input']
        
        if isinstance(data, dict) and 'data' in data:
            data = data['data']
        
        try:
            if method == 'min_max':
                min_val = np.min(data)
                max_val = np.max(data)
                normalized = (data - min_val) / (max_val - min_val)
            elif method == 'z_score':
                mean = np.mean(data)
                std = np.std(data)
                normalized = (data - mean) / std
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
            
            return {
                'data': normalized,
                'method': method,
                'shape': normalized.shape
            }
        except Exception as e:
            raise RuntimeError(f"Error normalizing data: {str(e)}")
    
    def _execute_one_hot_encoding(self, params, inputs):
        """One-hot encode categorical data"""
        if 'input' not in inputs:
            raise ValueError("One-hot encoding component requires input data")
        
        data = inputs['input']
        if isinstance(data, dict) and 'labels' in data:
            data = data['labels']
        
        try:
            if self.framework == 'tensorflow':
                encoded = tf.one_hot(data, depth=int(params.get('num_classes', np.max(data) + 1)))
                if hasattr(encoded, 'numpy'):
                    encoded = encoded.numpy()
            else:
                num_classes = int(params.get('num_classes', np.max(data) + 1))
                encoded = np.zeros((len(data), num_classes))
                encoded[np.arange(len(data)), data] = 1
            
            return {
                'data': encoded,
                'shape': encoded.shape,
                'num_classes': encoded.shape[-1]
            }
        except Exception as e:
            raise RuntimeError(f"Error one-hot encoding data: {str(e)}")
    
    def _execute_train_test_split(self, params, inputs):
        """Split data into training and testing sets"""
        if 'input' not in inputs:
            raise ValueError("Train-test split component requires input data")
        
        data = inputs['input']
        test_size = float(params.get('test_size', 0.2))
        random_state = int(params.get('random_state', 42))
        
        try:
            from sklearn.model_selection import train_test_split
            
            if isinstance(data, dict):
                if 'data' in data and 'labels' in data:
                    # Handle data with separate features and labels
                    X = data['data']
                    y = data['labels']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    return {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'train_size': len(X_train),
                        'test_size': len(X_test)
                    }
                elif 'images' in data and 'labels' in data:
                    # Handle image data
                    X = data['images']
                    y = data['labels']
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    return {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'train_size': len(X_train),
                        'test_size': len(X_test)
                    }
                elif 'data' in data:
                    # Handle data without labels
                    X = data['data']
                    X_train, X_test = train_test_split(
                        X, test_size=test_size, random_state=random_state
                    )
                    return {
                        'X_train': X_train,
                        'X_test': X_test,
                        'train_size': len(X_train),
                        'test_size': len(X_test)
                    }
            
            # Handle raw numpy array
            train_data, test_data = train_test_split(
                data, test_size=test_size, random_state=random_state
            )
            return {
                'train_data': train_data,
                'test_data': test_data,
                'train_size': len(train_data),
                'test_size': len(test_data)
            }
        except Exception as e:
            raise RuntimeError(f"Error splitting data: {str(e)}")
    
    def _execute_image_resize(self, params, inputs):
        """Resize images"""
        if 'input' not in inputs:
            raise ValueError("Image resize component requires input data")
        
        data = inputs['input']
        if isinstance(data, dict) and 'images' in data:
            images = data['images']
        else:
            images = data
        
        width = int(params.get('width', 224))
        height = int(params.get('height', 224))
        
        try:
            from PIL import Image
            
            resized_images = []
            for img in images:
                pil_img = Image.fromarray(img)
                resized_img = pil_img.resize((width, height))
                resized_images.append(np.array(resized_img))
            
            resized_array = np.array(resized_images)
            
            # If original input was a dict with images, update it
            if isinstance(data, dict) and 'images' in data:
                result = data.copy()
                result['images'] = resized_array
                result['shape'] = resized_array.shape
                return result
            
            return {
                'images': resized_array,
                'shape': resized_array.shape
            }
        except Exception as e:
            raise RuntimeError(f"Error resizing images: {str(e)}")
