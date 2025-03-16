"""
TensorFlow Executor - Handles TensorFlow-specific model execution
"""
import tensorflow as tf
import numpy as np
import time
import traceback

class TFExecutor:
    """
    Executes TensorFlow components in ML workflows
    """
    def __init__(self):
        self.models = {}
        self.layers = {}
        self.optimizers = {}
        self.losses = {}
        
    def execute_tf_layer(self, component_type, params, inputs):
        """
        Execute a TensorFlow layer component
        
        Args:
            component_type (str): Type of layer
            params (dict): Layer parameters
            inputs (dict): Input data
            
        Returns:
            dict: Layer configuration and output if input data is provided
        """
        layer_config = self._create_tf_layer(component_type, params)
        
        # Store layer configuration
        layer_id = f"layer_{time.time()}"
        self.layers[layer_id] = layer_config
        
        # Process input data if available
        if 'input' in inputs:
            input_data = inputs['input']
            if isinstance(input_data, dict) and 'data' in input_data:
                input_data = input_data['data']
                
            # Convert to tensor if needed
            if not isinstance(input_data, tf.Tensor):
                input_data = tf.convert_to_tensor(input_data)
                
            # Apply layer to input
            output = layer_config['layer'](input_data)
            
            return {
                'layer_id': layer_id,
                'layer_type': component_type,
                'params': params,
                'output': output.numpy() if hasattr(output, 'numpy') else output,
                'output_shape': output.shape.as_list() if hasattr(output, 'shape') else None
            }
        
        return {
            'layer_id': layer_id,
            'layer_type': component_type,
            'params': params
        }
        
    def _create_tf_layer(self, layer_type, params):
        """
        Create a TensorFlow layer based on type and parameters
        
        Args:
            layer_type (str): Type of layer
            params (dict): Layer parameters
            
        Returns:
            dict: Layer configuration
        """
        if layer_type == 'dense':
            units = int(params.get('units', 10))
            activation = params.get('activation', None)
            use_bias = params.get('use_bias', 'true').lower() == 'true'
            
            layer = tf.keras.layers.Dense(
                units=units,
                activation=activation,
                use_bias=use_bias
            )
            
            return {
                'layer': layer,
                'type': 'dense',
                'params': {
                    'units': units,
                    'activation': activation,
                    'use_bias': use_bias
                }
            }
            
        elif layer_type == 'conv2d':
            filters = int(params.get('filters', 32))
            kernel_size = tuple(map(int, params.get('kernel_size', '3,3').split(',')))
            strides = tuple(map(int, params.get('strides', '1,1').split(',')))
            padding = params.get('padding', 'valid')
            activation = params.get('activation', None)
            
            layer = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                activation=activation
            )
            
            return {
                'layer': layer,
                'type': 'conv2d',
                'params': {
                    'filters': filters,
                    'kernel_size': kernel_size,
                    'strides': strides,
                    'padding': padding,
                    'activation': activation
                }
            }
            
        elif layer_type == 'maxpooling2d':
            pool_size = tuple(map(int, params.get('pool_size', '2,2').split(',')))
            strides = tuple(map(int, params.get('strides', '2,2').split(',')))
            padding = params.get('padding', 'valid')
            
            layer = tf.keras.layers.MaxPooling2D(
                pool_size=pool_size,
                strides=strides,
                padding=padding
            )
            
            return {
                'layer': layer,
                'type': 'maxpooling2d',
                'params': {
                    'pool_size': pool_size,
                    'strides': strides,
                    'padding': padding
                }
            }
            
        elif layer_type == 'flatten':
            layer = tf.keras.layers.Flatten()
            
            return {
                'layer': layer,
                'type': 'flatten',
                'params': {}
            }
            
        elif layer_type == 'dropout':
            rate = float(params.get('rate', 0.5))
            
            layer = tf.keras.layers.Dropout(rate=rate)
            
            return {
                'layer': layer,
                'type': 'dropout',
                'params': {
                    'rate': rate
                }
            }
            
        elif layer_type == 'batch_normalization':
            axis = int(params.get('axis', -1))
            momentum = float(params.get('momentum', 0.99))
            epsilon = float(params.get('epsilon', 0.001))
            
            layer = tf.keras.layers.BatchNormalization(
                axis=axis,
                momentum=momentum,
                epsilon=epsilon
            )
            
            return {
                'layer': layer,
                'type': 'batch_normalization',
                'params': {
                    'axis': axis,
                    'momentum': momentum,
                    'epsilon': epsilon
                }
            }
            
        else:
            raise ValueError(f"Unsupported TensorFlow layer type: {layer_type}")
    
    def execute_tf_activation(self, activation_type, params, inputs):
        """
        Execute a TensorFlow activation function
        
        Args:
            activation_type (str): Type of activation function
            params (dict): Activation parameters
            inputs (dict): Input data
            
        Returns:
            dict: Activation configuration and output if input data is provided
        """
        activation_config = self._create_tf_activation(activation_type, params)
        
        # Process input data if available
        if 'input' in inputs:
            input_data = inputs['input']
            if isinstance(input_data, dict) and 'data' in input_data:
                input_data = input_data['data']
                
            # Convert to tensor if needed
            if not isinstance(input_data, tf.Tensor):
                input_data = tf.convert_to_tensor(input_data)
                
            # Apply activation to input
            output = activation_config['activation'](input_data)
            
            return {
                'activation_type': activation_type,
                'params': params,
                'output': output.numpy() if hasattr(output, 'numpy') else output,
                'output_shape': output.shape.as_list() if hasattr(output, 'shape') else None
            }
        
        return {
            'activation_type': activation_type,
            'params': params
        }
        
    def _create_tf_activation(self, activation_type, params):
        """
        Create a TensorFlow activation function
        
        Args:
            activation_type (str): Type of activation function
            params (dict): Activation parameters
            
        Returns:
            dict: Activation configuration
        """
        if activation_type == 'relu':
            activation = tf.keras.activations.relu
            
        elif activation_type == 'sigmoid':
            activation = tf.keras.activations.sigmoid
            
        elif activation_type == 'tanh':
            activation = tf.keras.activations.tanh
            
        elif activation_type == 'softmax':
            axis = int(params.get('axis', -1))
            activation = lambda x: tf.keras.activations.softmax(x, axis=axis)
            
        elif activation_type == 'leaky_relu':
            alpha = float(params.get('alpha', 0.3))
            activation = lambda x: tf.keras.activations.relu(x, alpha=alpha)
            
        else:
            raise ValueError(f"Unsupported TensorFlow activation type: {activation_type}")
            
        return {
            'activation': activation,
            'type': activation_type,
            'params': params
        }
    
    def execute_tf_optimizer(self, optimizer_type, params, inputs):
        """
        Execute a TensorFlow optimizer component
        
        Args:
            optimizer_type (str): Type of optimizer
            params (dict): Optimizer parameters
            inputs (dict): Input data (unused for optimizers)
            
        Returns:
            dict: Optimizer configuration
        """
        optimizer_config = self._create_tf_optimizer(optimizer_type, params)
        
        # Store optimizer configuration
        optimizer_id = f"optimizer_{time.time()}"
        self.optimizers[optimizer_id] = optimizer_config
        
        return {
            'optimizer_id': optimizer_id,
            'optimizer_type': optimizer_type,
            'params': params
        }
        
    def _create_tf_optimizer(self, optimizer_type, params):
        """
        Create a TensorFlow optimizer
        
        Args:
            optimizer_type (str): Type of optimizer
            params (dict): Optimizer parameters
            
        Returns:
            dict: Optimizer configuration
        """
        if optimizer_type == 'adam':
            learning_rate = float(params.get('learning_rate', 0.001))
            beta_1 = float(params.get('beta_1', 0.9))
            beta_2 = float(params.get('beta_2', 0.999))
            epsilon = float(params.get('epsilon', 1e-7))
            
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon
            )
            
            return {
                'optimizer': optimizer,
                'type': 'adam',
                'params': {
                    'learning_rate': learning_rate,
                    'beta_1': beta_1,
                    'beta_2': beta_2,
                    'epsilon': epsilon
                }
            }
            
        elif optimizer_type == 'sgd':
            learning_rate = float(params.get('learning_rate', 0.01))
            momentum = float(params.get('momentum', 0.0))
            nesterov = params.get('nesterov', 'false').lower() == 'true'
            
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=momentum,
                nesterov=nesterov
            )
            
            return {
                'optimizer': optimizer,
                'type': 'sgd',
                'params': {
                    'learning_rate': learning_rate,
                    'momentum': momentum,
                    'nesterov': nesterov
                }
            }
            
        elif optimizer_type == 'rmsprop':
            learning_rate = float(params.get('learning_rate', 0.001))
            rho = float(params.get('rho', 0.9))
            momentum = float(params.get('momentum', 0.0))
            epsilon = float(params.get('epsilon', 1e-7))
            
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                rho=rho,
                momentum=momentum,
                epsilon=epsilon
            )
            
            return {
                'optimizer': optimizer,
                'type': 'rmsprop',
                'params': {
                    'learning_rate': learning_rate,
                    'rho': rho,
                    'momentum': momentum,
                    'epsilon': epsilon
                }
            }
            
        else:
            raise ValueError(f"Unsupported TensorFlow optimizer type: {optimizer_type}")
    
    def execute_tf_loss(self, loss_type, params, inputs):
        """
        Execute a TensorFlow loss function component
        
        Args:
            loss_type (str): Type of loss function
            params (dict): Loss function parameters
            inputs (dict): Input data
            
        Returns:
            dict: Loss function configuration and output if input data is provided
        """
        loss_config = self._create_tf_loss(loss_type, params)
        
        # Store loss configuration
        loss_id = f"loss_{time.time()}"
        self.losses[loss_id] = loss_config
        
        # Process input data if available
        if 'y_true' in inputs and 'y_pred' in inputs:
            y_true = inputs['y_true']
            y_pred = inputs['y_pred']
            
            if isinstance(y_true, dict) and 'data' in y_true:
                y_true = y_true['data']
            if isinstance(y_pred, dict) and 'data' in y_pred:
                y_pred = y_pred['data']
                
            # Convert to tensors if needed
            if not isinstance(y_true, tf.Tensor):
                y_true = tf.convert_to_tensor(y_true)
            if not isinstance(y_pred, tf.Tensor):
                y_pred = tf.convert_to_tensor(y_pred)
                
            # Calculate loss
            loss_value = loss_config['loss'](y_true, y_pred)
            
            return {
                'loss_id': loss_id,
                'loss_type': loss_type,
                'params': params,
                'loss_value': loss_value.numpy() if hasattr(loss_value, 'numpy') else loss_value
            }
        
        return {
            'loss_id': loss_id,
            'loss_type': loss_type,
            'params': params
        }
        
    def _create_tf_loss(self, loss_type, params):
        """
        Create a TensorFlow loss function
        
        Args:
            loss_type (str): Type of loss function
            params (dict): Loss function parameters
            
        Returns:
            dict: Loss function configuration
        """
        if loss_type == 'categorical_crossentropy':
            from_logits = params.get('from_logits', 'false').lower() == 'true'
            label_smoothing = float(params.get('label_smoothing', 0.0))
            
            loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=from_logits,
                label_smoothing=label_smoothing
            )
            
            return {
                'loss': loss,
                'type': 'categorical_crossentropy',
                'params': {
                    'from_logits': from_logits,
                    'label_smoothing': label_smoothing
                }
            }
            
        elif loss_type == 'binary_crossentropy':
            from_logits = params.get('from_logits', 'false').lower() == 'true'
            label_smoothing = float(params.get('label_smoothing', 0.0))
            
            loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=from_logits,
                label_smoothing=label_smoothing
            )
            
            return {
                'loss': loss,
                'type': 'binary_crossentropy',
                'params': {
                    'from_logits': from_logits,
                    'label_smoothing': label_smoothing
                }
            }
            
        elif loss_type == 'mse':
            loss = tf.keras.losses.MeanSquaredError()
            
            return {
                'loss': loss,
                'type': 'mse',
                'params': {}
            }
            
        elif loss_type == 'mae':
            loss = tf.keras.losses.MeanAbsoluteError()
            
            return {
                'loss': loss,
                'type': 'mae',
                'params': {}
            }
            
        else:
            raise ValueError(f"Unsupported TensorFlow loss type: {loss_type}")
    
    def execute_tf_model(self, model_type, params, inputs):
        """
        Execute a TensorFlow model component
        
        Args:
            model_type (str): Type of model
            params (dict): Model parameters
            inputs (dict): Input data including layers
            
        Returns:
            dict: Model configuration
        """
        if model_type == 'sequential':
            # Get layers from inputs
            layers = []
            for key, value in inputs.items():
                if key.startswith('layer_'):
                    if isinstance(value, dict) and 'layer' in value:
                        layers.append(value['layer'])
            
            # Create sequential model
            model = tf.keras.Sequential(layers)
            
            # Store model
            model_id = f"model_{time.time()}"
            self.models[model_id] = {
                'model': model,
                'type': 'sequential',
                'params': params
            }
            
            return {
                'model_id': model_id,
                'model_type': 'sequential',
                'params': params,
                'layer_count': len(layers)
            }
            
        elif model_type == 'functional':
            # This is a simplified implementation
            # A complete implementation would handle complex layer connections
            raise NotImplementedError("Functional API model not fully implemented yet")
            
        else:
            raise ValueError(f"Unsupported TensorFlow model type: {model_type}")
    
    def execute_tf_training(self, training_type, params, inputs):
        """
        Execute a TensorFlow training component
        
        Args:
            training_type (str): Type of training
            params (dict): Training parameters
            inputs (dict): Input data including model, optimizer, loss, and data
            
        Returns:
            dict: Training results
        """
        if 'model' not in inputs:
            raise ValueError("Training component requires a model input")
        if 'optimizer' not in inputs:
            raise ValueError("Training component requires an optimizer input")
        if 'loss' not in inputs:
            raise ValueError("Training component requires a loss input")
        if 'X_train' not in inputs:
            raise ValueError("Training component requires training data (X_train)")
        if 'y_train' not in inputs:
            raise ValueError("Training component requires training labels (y_train)")
        
        # Get model
        model_input = inputs['model']
        if isinstance(model_input, dict) and 'model_id' in model_input:
            model_id = model_input['model_id']
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            model = self.models[model_id]['model']
        else:
            raise ValueError("Invalid model input")
        
        # Get optimizer
        optimizer_input = inputs['optimizer']
        if isinstance(optimizer_input, dict) and 'optimizer_id' in optimizer_input:
            optimizer_id = optimizer_input['optimizer_id']
            if optimizer_id not in self.optimizers:
                raise ValueError(f"Optimizer {optimizer_id} not found")
            optimizer = self.optimizers[optimizer_id]['optimizer']
        else:
            raise ValueError("Invalid optimizer input")
        
        # Get loss
        loss_input = inputs['loss']
        if isinstance(loss_input, dict) and 'loss_id' in loss_input:
            loss_id = loss_input['loss_id']
            if loss_id not in self.losses:
                raise ValueError(f"Loss {loss_id} not found")
            loss = self.losses[loss_id]['loss']
        else:
            raise ValueError("Invalid loss input")
        
        # Get training data
        X_train = inputs['X_train']
        y_train = inputs['y_train']
        
        if isinstance(X_train, dict) and 'data' in X_train:
            X_train = X_train['data']
        if isinstance(y_train, dict) and 'data' in y_train:
            y_train = y_train['data']
        
        # Get validation data if available
        validation_data = None
        if 'X_val' in inputs and 'y_val' in inputs:
            X_val = inputs['X_val']
            y_val = inputs['y_val']
            
            if isinstance(X_val, dict) and 'data' in X_val:
                X_val = X_val['data']
            if isinstance(y_val, dict) and 'data' in y_val:
                y_val = y_val['data']
                
            validation_data = (X_val, y_val)
        
        # Get training parameters
        epochs = int(params.get('epochs', 10))
        batch_size = int(params.get('batch_size', 32))
        verbose = int(params.get('verbose', 1))
        
        # Compile model
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=validation_data
        )
        
        # Convert history to dict
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        
        return {
            'model_id': model_id,
            'history': history_dict,
            'epochs': epochs,
            'batch_size': batch_size
        }
    
    def execute_tf_evaluation(self, evaluation_type, params, inputs):
        """
        Execute a TensorFlow evaluation component
        
        Args:
            evaluation_type (str): Type of evaluation
            params (dict): Evaluation parameters
            inputs (dict): Input data including model and test data
            
        Returns:
            dict: Evaluation results
        """
        if 'model' not in inputs:
            raise ValueError("Evaluation component requires a model input")
        if 'X_test' not in inputs:
            raise ValueError("Evaluation component requires test data (X_test)")
        if 'y_test' not in inputs:
            raise ValueError("Evaluation component requires test labels (y_test)")
        
        # Get model
        model_input = inputs['model']
        if isinstance(model_input, dict) and 'model_id' in model_input:
            model_id = model_input['model_id']
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            model = self.models[model_id]['model']
        else:
            raise ValueError("Invalid model input")
        
        # Get test data
        X_test = inputs['X_test']
        y_test = inputs['y_test']
        
        if isinstance(X_test, dict) and 'data' in X_test:
            X_test = X_test['data']
        if isinstance(y_test, dict) and 'data' in y_test:
            y_test = y_test['data']
        
        # Get evaluation parameters
        batch_size = int(params.get('batch_size', 32))
        verbose = int(params.get('verbose', 1))
        
        # Evaluate model
        evaluation = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
        
        # Get predictions
        predictions = model.predict(X_test, batch_size=batch_size, verbose=verbose)
        
        # Convert predictions to class indices if needed
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predicted_classes = np.argmax(predictions, axis=1)
        else:
            predicted_classes = (predictions > 0.5).astype(int)
        
        # Convert true labels to class indices if needed
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            true_classes = np.argmax(y_test, axis=1)
        else:
            true_classes = y_test
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        try:
            accuracy = float(accuracy_score(true_classes, predicted_classes))
            precision = float(precision_score(true_classes, predicted_classes, average='weighted', zero_division=0))
            recall = float(recall_score(true_classes, predicted_classes, average='weighted', zero_division=0))
            f1 = float(f1_score(true_classes, predicted_classes, average='weighted', zero_division=0))
            conf_matrix = confusion_matrix(true_classes, predicted_classes).tolist()
        except Exception as e:
            # Handle errors in metric calculation
            error_msg = f"Error calculating metrics: {str(e)}"
            traceback.print_exc()
            return {
                'model_id': model_id,
                'error': error_msg
            }
        
        return {
            'model_id': model_id,
            'loss': float(evaluation[0]),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'predictions': predictions.tolist(),
            'predicted_classes': predicted_classes.tolist()
        }
