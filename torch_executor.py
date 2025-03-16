"""
PyTorch Executor - Handles PyTorch-specific model execution
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import traceback

class TorchExecutor:
    """
    Executes PyTorch components in ML workflows
    """
    def __init__(self):
        self.models = {}
        self.layers = {}
        self.optimizers = {}
        self.losses = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def execute_torch_layer(self, component_type, params, inputs):
        """
        Execute a PyTorch layer component
        
        Args:
            component_type (str): Type of layer
            params (dict): Layer parameters
            inputs (dict): Input data
            
        Returns:
            dict: Layer configuration and output if input data is provided
        """
        layer_config = self._create_torch_layer(component_type, params)
        
        # Store layer configuration
        layer_id = f"layer_{time.time()}"
        self.layers[layer_id] = layer_config
        
        # Process input data if available
        if 'input' in inputs:
            input_data = inputs['input']
            if isinstance(input_data, dict) and 'data' in input_data:
                input_data = input_data['data']
                
            # Convert to tensor if needed
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)
                
            # Apply layer to input
            with torch.no_grad():
                output = layer_config['layer'](input_data)
            
            return {
                'layer_id': layer_id,
                'layer_type': component_type,
                'params': params,
                'output': output.cpu().numpy() if hasattr(output, 'cpu') else output,
                'output_shape': list(output.shape) if hasattr(output, 'shape') else None
            }
        
        return {
            'layer_id': layer_id,
            'layer_type': component_type,
            'params': params
        }
        
    def _create_torch_layer(self, layer_type, params):
        """
        Create a PyTorch layer based on type and parameters
        
        Args:
            layer_type (str): Type of layer
            params (dict): Layer parameters
            
        Returns:
            dict: Layer configuration
        """
        if layer_type == 'linear':
            in_features = int(params.get('in_features', 10))
            out_features = int(params.get('out_features', 10))
            bias = params.get('bias', 'true').lower() == 'true'
            
            layer = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias
            ).to(self.device)
            
            return {
                'layer': layer,
                'type': 'linear',
                'params': {
                    'in_features': in_features,
                    'out_features': out_features,
                    'bias': bias
                }
            }
            
        elif layer_type == 'conv2d':
            in_channels = int(params.get('in_channels', 3))
            out_channels = int(params.get('out_channels', 32))
            kernel_size = tuple(map(int, params.get('kernel_size', '3,3').split(',')))
            stride = tuple(map(int, params.get('stride', '1,1').split(',')))
            padding = tuple(map(int, params.get('padding', '0,0').split(',')))
            
            layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ).to(self.device)
            
            return {
                'layer': layer,
                'type': 'conv2d',
                'params': {
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding
                }
            }
            
        elif layer_type == 'maxpool2d':
            kernel_size = tuple(map(int, params.get('kernel_size', '2,2').split(',')))
            stride = tuple(map(int, params.get('stride', '2,2').split(',')))
            padding = tuple(map(int, params.get('padding', '0,0').split(',')))
            
            layer = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ).to(self.device)
            
            return {
                'layer': layer,
                'type': 'maxpool2d',
                'params': {
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding
                }
            }
            
        elif layer_type == 'flatten':
            layer = nn.Flatten().to(self.device)
            
            return {
                'layer': layer,
                'type': 'flatten',
                'params': {}
            }
            
        elif layer_type == 'dropout':
            p = float(params.get('p', 0.5))
            
            layer = nn.Dropout(p=p).to(self.device)
            
            return {
                'layer': layer,
                'type': 'dropout',
                'params': {
                    'p': p
                }
            }
            
        elif layer_type == 'batch_norm2d':
            num_features = int(params.get('num_features', 32))
            eps = float(params.get('eps', 1e-5))
            momentum = float(params.get('momentum', 0.1))
            
            layer = nn.BatchNorm2d(
                num_features=num_features,
                eps=eps,
                momentum=momentum
            ).to(self.device)
            
            return {
                'layer': layer,
                'type': 'batch_norm2d',
                'params': {
                    'num_features': num_features,
                    'eps': eps,
                    'momentum': momentum
                }
            }
            
        else:
            raise ValueError(f"Unsupported PyTorch layer type: {layer_type}")
    
    def execute_torch_activation(self, activation_type, params, inputs):
        """
        Execute a PyTorch activation function
        
        Args:
            activation_type (str): Type of activation function
            params (dict): Activation parameters
            inputs (dict): Input data
            
        Returns:
            dict: Activation configuration and output if input data is provided
        """
        activation_config = self._create_torch_activation(activation_type, params)
        
        # Process input data if available
        if 'input' in inputs:
            input_data = inputs['input']
            if isinstance(input_data, dict) and 'data' in input_data:
                input_data = input_data['data']
                
            # Convert to tensor if needed
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)
                
            # Apply activation to input
            with torch.no_grad():
                output = activation_config['activation'](input_data)
            
            return {
                'activation_type': activation_type,
                'params': params,
                'output': output.cpu().numpy() if hasattr(output, 'cpu') else output,
                'output_shape': list(output.shape) if hasattr(output, 'shape') else None
            }
        
        return {
            'activation_type': activation_type,
            'params': params
        }
        
    def _create_torch_activation(self, activation_type, params):
        """
        Create a PyTorch activation function
        
        Args:
            activation_type (str): Type of activation function
            params (dict): Activation parameters
            
        Returns:
            dict: Activation configuration
        """
        if activation_type == 'relu':
            activation = nn.ReLU().to(self.device)
            
        elif activation_type == 'sigmoid':
            activation = nn.Sigmoid().to(self.device)
            
        elif activation_type == 'tanh':
            activation = nn.Tanh().to(self.device)
            
        elif activation_type == 'softmax':
            dim = int(params.get('dim', -1))
            activation = nn.Softmax(dim=dim).to(self.device)
            
        elif activation_type == 'leaky_relu':
            negative_slope = float(params.get('negative_slope', 0.01))
            activation = nn.LeakyReLU(negative_slope=negative_slope).to(self.device)
            
        else:
            raise ValueError(f"Unsupported PyTorch activation type: {activation_type}")
            
        return {
            'activation': activation,
            'type': activation_type,
            'params': params
        }
    
    def execute_torch_optimizer(self, optimizer_type, params, inputs):
        """
        Execute a PyTorch optimizer component
        
        Args:
            optimizer_type (str): Type of optimizer
            params (dict): Optimizer parameters
            inputs (dict): Input data including model parameters
            
        Returns:
            dict: Optimizer configuration
        """
        # Get model parameters if available
        model_params = None
        if 'model' in inputs:
            model_input = inputs['model']
            if isinstance(model_input, dict) and 'model_id' in model_input:
                model_id = model_input['model_id']
                if model_id in self.models:
                    model = self.models[model_id]['model']
                    model_params = model.parameters()
        
        # Create optimizer
        optimizer_config = self._create_torch_optimizer(optimizer_type, params, model_params)
        
        # Store optimizer configuration
        optimizer_id = f"optimizer_{time.time()}"
        self.optimizers[optimizer_id] = optimizer_config
        
        return {
            'optimizer_id': optimizer_id,
            'optimizer_type': optimizer_type,
            'params': params
        }
        
    def _create_torch_optimizer(self, optimizer_type, params, model_params=None):
        """
        Create a PyTorch optimizer
        
        Args:
            optimizer_type (str): Type of optimizer
            params (dict): Optimizer parameters
            model_params (iterator): Model parameters to optimize
            
        Returns:
            dict: Optimizer configuration
        """
        if model_params is None:
            # Create a dummy parameter if no model parameters are provided
            dummy = nn.Parameter(torch.zeros(1)).to(self.device)
            model_params = [dummy]
        
        if optimizer_type == 'adam':
            lr = float(params.get('lr', 0.001))
            betas = tuple(map(float, params.get('betas', '0.9,0.999').split(',')))
            eps = float(params.get('eps', 1e-8))
            weight_decay = float(params.get('weight_decay', 0))
            
            optimizer = optim.Adam(
                model_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
            
            return {
                'optimizer': optimizer,
                'type': 'adam',
                'params': {
                    'lr': lr,
                    'betas': betas,
                    'eps': eps,
                    'weight_decay': weight_decay
                }
            }
            
        elif optimizer_type == 'sgd':
            lr = float(params.get('lr', 0.01))
            momentum = float(params.get('momentum', 0.0))
            weight_decay = float(params.get('weight_decay', 0))
            nesterov = params.get('nesterov', 'false').lower() == 'true'
            
            optimizer = optim.SGD(
                model_params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov
            )
            
            return {
                'optimizer': optimizer,
                'type': 'sgd',
                'params': {
                    'lr': lr,
                    'momentum': momentum,
                    'weight_decay': weight_decay,
                    'nesterov': nesterov
                }
            }
            
        elif optimizer_type == 'rmsprop':
            lr = float(params.get('lr', 0.01))
            alpha = float(params.get('alpha', 0.99))
            eps = float(params.get('eps', 1e-8))
            weight_decay = float(params.get('weight_decay', 0))
            momentum = float(params.get('momentum', 0))
            
            optimizer = optim.RMSprop(
                model_params,
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum
            )
            
            return {
                'optimizer': optimizer,
                'type': 'rmsprop',
                'params': {
                    'lr': lr,
                    'alpha': alpha,
                    'eps': eps,
                    'weight_decay': weight_decay,
                    'momentum': momentum
                }
            }
            
        else:
            raise ValueError(f"Unsupported PyTorch optimizer type: {optimizer_type}")
    
    def execute_torch_loss(self, loss_type, params, inputs):
        """
        Execute a PyTorch loss function component
        
        Args:
            loss_type (str): Type of loss function
            params (dict): Loss function parameters
            inputs (dict): Input data
            
        Returns:
            dict: Loss function configuration and output if input data is provided
        """
        loss_config = self._create_torch_loss(loss_type, params)
        
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
            if not isinstance(y_true, torch.Tensor):
                y_true = torch.tensor(y_true, dtype=torch.float32).to(self.device)
            if not isinstance(y_pred, torch.Tensor):
                y_pred = torch.tensor(y_pred, dtype=torch.float32).to(self.device)
                
            # Calculate loss
            with torch.no_grad():
                loss_value = loss_config['loss'](y_pred, y_true)
            
            return {
                'loss_id': loss_id,
                'loss_type': loss_type,
                'params': params,
                'loss_value': loss_value.item() if hasattr(loss_value, 'item') else loss_value
            }
        
        return {
            'loss_id': loss_id,
            'loss_type': loss_type,
            'params': params
        }
        
    def _create_torch_loss(self, loss_type, params):
        """
        Create a PyTorch loss function
        
        Args:
            loss_type (str): Type of loss function
            params (dict): Loss function parameters
            
        Returns:
            dict: Loss function configuration
        """
        if loss_type == 'cross_entropy':
            weight = None
            if 'weight' in params:
                weight_str = params.get('weight')
                weight = torch.tensor([float(w) for w in weight_str.split(',')], dtype=torch.float32).to(self.device)
                
            reduction = params.get('reduction', 'mean')
            label_smoothing = float(params.get('label_smoothing', 0.0))
            
            loss = nn.CrossEntropyLoss(
                weight=weight,
                reduction=reduction,
                label_smoothing=label_smoothing
            ).to(self.device)
            
            return {
                'loss': loss,
                'type': 'cross_entropy',
                'params': {
                    'weight': weight,
                    'reduction': reduction,
                    'label_smoothing': label_smoothing
                }
            }
            
        elif loss_type == 'bce':
            weight = None
            if 'weight' in params:
                weight_str = params.get('weight')
                weight = torch.tensor([float(w) for w in weight_str.split(',')], dtype=torch.float32).to(self.device)
                
            reduction = params.get('reduction', 'mean')
            
            loss = nn.BCELoss(
                weight=weight,
                reduction=reduction
            ).to(self.device)
            
            return {
                'loss': loss,
                'type': 'bce',
                'params': {
                    'weight': weight,
                    'reduction': reduction
                }
            }
            
        elif loss_type == 'mse':
            reduction = params.get('reduction', 'mean')
            
            loss = nn.MSELoss(reduction=reduction).to(self.device)
            
            return {
                'loss': loss,
                'type': 'mse',
                'params': {
                    'reduction': reduction
                }
            }
            
        elif loss_type == 'l1':
            reduction = params.get('reduction', 'mean')
            
            loss = nn.L1Loss(reduction=reduction).to(self.device)
            
            return {
                'loss': loss,
                'type': 'l1',
                'params': {
                    'reduction': reduction
                }
            }
            
        else:
            raise ValueError(f"Unsupported PyTorch loss type: {loss_type}")
    
    def execute_torch_model(self, model_type, params, inputs):
        """
        Execute a PyTorch model component
        
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
            model = nn.Sequential(*layers).to(self.device)
            
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
            
        elif model_type == 'custom':
            # This is a placeholder for custom model implementation
            raise NotImplementedError("Custom model not fully implemented yet")
            
        else:
            raise ValueError(f"Unsupported PyTorch model type: {model_type}")
    
    def execute_torch_training(self, training_type, params, inputs):
        """
        Execute a PyTorch training component
        
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
            loss_fn = self.losses[loss_id]['loss']
        else:
            raise ValueError("Invalid loss input")
        
        # Get training data
        X_train = inputs['X_train']
        y_train = inputs['y_train']
        
        if isinstance(X_train, dict) and 'data' in X_train:
            X_train = X_train['data']
        if isinstance(y_train, dict) and 'data' in y_train:
            y_train = y_train['data']
        
        # Convert to tensors if needed
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        # Get validation data if available
        X_val = None
        y_val = None
        if 'X_val' in inputs and 'y_val' in inputs:
            X_val = inputs['X_val']
            y_val = inputs['y_val']
            
            if isinstance(X_val, dict) and 'data' in X_val:
                X_val = X_val['data']
            if isinstance(y_val, dict) and 'data' in y_val:
                y_val = y_val['data']
                
            # Convert to tensors if needed
            if not isinstance(X_val, torch.Tensor):
                X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            if not isinstance(y_val, torch.Tensor):
                y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        
        # Get training parameters
        epochs = int(params.get('epochs', 10))
        batch_size = int(params.get('batch_size', 32))
        
        # Create data loader
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        history = {
            'loss': [],
            'accuracy': []
        }
        
        if X_val is not None and y_val is not None:
            history['val_loss'] = []
            history['val_accuracy'] = []
        
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track statistics
                epoch_loss += loss.item()
                
                # Calculate accuracy
                if outputs.shape[1] > 1:  # Multi-class classification
                    _, predicted = torch.max(outputs.data, 1)
                    if batch_y.dim() > 1:  # One-hot encoded
                        _, targets = torch.max(batch_y.data, 1)
                    else:
                        targets = batch_y
                else:  # Binary classification
                    predicted = (outputs.data > 0.5).float()
                    targets = batch_y
                    
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            # Calculate epoch statistics
            epoch_loss /= len(train_loader)
            epoch_accuracy = correct / total
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)
            
            # Validation
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_loss = loss_fn(val_outputs, y_val).item()
                    
                    # Calculate validation accuracy
                    if val_outputs.shape[1] > 1:  # Multi-class classification
                        _, val_predicted = torch.max(val_outputs.data, 1)
                        if y_val.dim() > 1:  # One-hot encoded
                            _, val_targets = torch.max(y_val.data, 1)
                        else:
                            val_targets = y_val
                    else:  # Binary classification
                        val_predicted = (val_outputs.data > 0.5).float()
                        val_targets = y_val
                        
                    val_correct = (val_predicted == val_targets).sum().item()
                    val_accuracy = val_correct / y_val.size(0)
                    
                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_accuracy)
                
                model.train()
        
        return {
            'model_id': model_id,
            'history': history,
            'epochs': epochs,
            'batch_size': batch_size
        }
    
    def execute_torch_evaluation(self, evaluation_type, params, inputs):
        """
        Execute a PyTorch evaluation component
        
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
        
        # Get loss function if available
        loss_fn = None
        if 'loss' in inputs:
            loss_input = inputs['loss']
            if isinstance(loss_input, dict) and 'loss_id' in loss_input:
                loss_id = loss_input['loss_id']
                if loss_id in self.losses:
                    loss_fn = self.losses[loss_id]['loss']
        
        # Get test data
        X_test = inputs['X_test']
        y_test = inputs['y_test']
        
        if isinstance(X_test, dict) and 'data' in X_test:
            X_test = X_test['data']
        if isinstance(y_test, dict) and 'data' in y_test:
            y_test = y_test['data']
        
        # Convert to tensors if needed
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            # Get predictions
            outputs = model(X_test)
            
            # Calculate loss if loss function is available
            loss = None
            if loss_fn is not None:
                loss = loss_fn(outputs, y_test).item()
            
            # Convert outputs to numpy for metrics calculation
            predictions = outputs.cpu().numpy()
            
            # Convert predictions to class indices if needed
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                predicted_classes = np.argmax(predictions, axis=1)
            else:
                predicted_classes = (predictions > 0.5).astype(int)
            
            # Convert true labels to class indices if needed
            true_classes = y_test.cpu().numpy()
            if len(true_classes.shape) > 1 and true_classes.shape[1] > 1:
                true_classes = np.argmax(true_classes, axis=1)
        
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
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'predictions': predictions.tolist(),
            'predicted_classes': predicted_classes.tolist()
        }
