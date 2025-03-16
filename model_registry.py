"""
Model Registry - Manages trained models for ML Sandbox
"""
import os
import json
import pickle
import uuid
import traceback
import shutil
from datetime import datetime
import tensorflow as tf
import torch

class ModelRegistry:
    """
    Manages trained models, including saving, loading, and versioning
    """
    def __init__(self, models_dir="models"):
        """
        Initialize the model registry
        
        Args:
            models_dir (str): Directory to store models
        """
        self.models_dir = models_dir
        self.model_metadata = {}
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "tensorflow"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "pytorch"), exist_ok=True)
        
        # Load existing metadata
        self._load_metadata()
    
    def register_model(self, model, framework, name, description="", tags=None, metrics=None, 
                      dataset_id=None, workflow_id=None):
        """
        Register a trained model
        
        Args:
            model: Trained model (TensorFlow or PyTorch)
            framework (str): Model framework ('tensorflow' or 'pytorch')
            name (str): Model name
            description (str): Model description
            tags (list): List of tags
            metrics (dict): Model performance metrics
            dataset_id (str): ID of the dataset used for training
            workflow_id (str): ID of the workflow used for training
            
        Returns:
            dict: Model information
        """
        # Validate framework
        if framework not in ["tensorflow", "pytorch"]:
            raise ValueError("Framework must be 'tensorflow' or 'pytorch'")
        
        # Generate model ID
        model_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, framework, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        try:
            if framework == "tensorflow":
                model_path = os.path.join(model_dir, "model")
                model.save(model_path)
            elif framework == "pytorch":
                model_path = os.path.join(model_dir, "model.pt")
                torch.save(model.state_dict(), model_path)
        except Exception as e:
            # Clean up directory if saving fails
            shutil.rmtree(model_dir, ignore_errors=True)
            error_msg = f"Error saving model: {str(e)}"
            traceback.print_exc()
            raise RuntimeError(error_msg)
        
        # Create model metadata
        model_info = {
            "id": model_id,
            "name": name,
            "description": description,
            "framework": framework,
            "created_at": created_at,
            "updated_at": created_at,
            "version": 1,
            "tags": tags or [],
            "metrics": metrics or {},
            "dataset_id": dataset_id,
            "workflow_id": workflow_id,
            "model_path": model_path
        }
        
        # Add model architecture info if possible
        if framework == "tensorflow":
            model_info["architecture"] = {
                "layers": [layer.name for layer in model.layers],
                "input_shape": model.input_shape[1:] if hasattr(model, 'input_shape') else None,
                "output_shape": model.output_shape[1:] if hasattr(model, 'output_shape') else None
            }
        elif framework == "pytorch":
            # Get model summary as string
            model_info["architecture"] = {
                "summary": str(model)
            }
        
        # Save metadata
        self.model_metadata[model_id] = model_info
        self._save_metadata()
        
        return model_info
    
    def load_model(self, model_id):
        """
        Load a model from the registry
        
        Args:
            model_id (str): ID of the model to load
            
        Returns:
            tuple: (model, model_info)
        """
        if model_id not in self.model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.model_metadata[model_id]
        model_path = model_info["model_path"]
        framework = model_info["framework"]
        
        try:
            if framework == "tensorflow":
                model = tf.keras.models.load_model(model_path)
            elif framework == "pytorch":
                # Need model class to load PyTorch model
                # This is a placeholder - in real usage, you'd need to provide the model class
                raise NotImplementedError(
                    "PyTorch model loading requires model class. Use load_pytorch_model instead."
                )
            else:
                raise ValueError(f"Unsupported framework: {framework}")
            
            return model, model_info
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            traceback.print_exc()
            raise RuntimeError(error_msg)
    
    def load_pytorch_model(self, model_id, model_class, *args, **kwargs):
        """
        Load a PyTorch model from the registry
        
        Args:
            model_id (str): ID of the model to load
            model_class: PyTorch model class
            *args, **kwargs: Arguments to pass to model class constructor
            
        Returns:
            tuple: (model, model_info)
        """
        if model_id not in self.model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.model_metadata[model_id]
        model_path = model_info["model_path"]
        framework = model_info["framework"]
        
        if framework != "pytorch":
            raise ValueError(f"Model {model_id} is not a PyTorch model")
        
        try:
            # Create model instance
            model = model_class(*args, **kwargs)
            
            # Load state dict
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set to evaluation mode
            
            return model, model_info
        except Exception as e:
            error_msg = f"Error loading PyTorch model: {str(e)}"
            traceback.print_exc()
            raise RuntimeError(error_msg)
    
    def list_models(self, framework=None, tags=None):
        """
        List models in the registry
        
        Args:
            framework (str): Filter by framework
            tags (list): Filter by tags
            
        Returns:
            list: List of model metadata
        """
        models = []
        
        for model_id, info in self.model_metadata.items():
            # Apply filters
            if framework and info["framework"] != framework:
                continue
            
            if tags and not all(tag in info["tags"] for tag in tags):
                continue
            
            # Add to results
            models.append({
                "id": model_id,
                "name": info["name"],
                "framework": info["framework"],
                "version": info["version"],
                "created_at": info["created_at"],
                "tags": info["tags"],
                "metrics": info["metrics"]
            })
        
        return models
    
    def get_model_info(self, model_id):
        """
        Get model information
        
        Args:
            model_id (str): ID of the model
            
        Returns:
            dict: Model information
        """
        if model_id not in self.model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        return self.model_metadata[model_id]
    
    def update_model_info(self, model_id, updates):
        """
        Update model information
        
        Args:
            model_id (str): ID of the model
            updates (dict): Updates to apply
            
        Returns:
            dict: Updated model information
        """
        if model_id not in self.model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.model_metadata[model_id]
        
        # Apply updates
        for key, value in updates.items():
            if key in ["name", "description", "tags", "metrics"]:
                model_info[key] = value
        
        # Update timestamp
        model_info["updated_at"] = datetime.now().isoformat()
        
        # Save metadata
        self._save_metadata()
        
        return model_info
    
    def create_model_version(self, model_id, model, metrics=None):
        """
        Create a new version of a model
        
        Args:
            model_id (str): ID of the model
            model: New model version
            metrics (dict): Model performance metrics
            
        Returns:
            dict: New model version information
        """
        if model_id not in self.model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.model_metadata[model_id]
        framework = model_info["framework"]
        
        # Create new version
        new_version = model_info["version"] + 1
        
        # Create version directory
        model_dir = os.path.join(self.models_dir, framework, model_id, f"v{new_version}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        try:
            if framework == "tensorflow":
                model_path = os.path.join(model_dir, "model")
                model.save(model_path)
            elif framework == "pytorch":
                model_path = os.path.join(model_dir, "model.pt")
                torch.save(model.state_dict(), model_path)
        except Exception as e:
            # Clean up directory if saving fails
            shutil.rmtree(model_dir, ignore_errors=True)
            error_msg = f"Error saving model version: {str(e)}"
            traceback.print_exc()
            raise RuntimeError(error_msg)
        
        # Update model metadata
        model_info["version"] = new_version
        model_info["updated_at"] = datetime.now().isoformat()
        model_info["model_path"] = model_path
        
        if metrics:
            model_info["metrics"] = metrics
        
        # Save metadata
        self._save_metadata()
        
        return model_info
    
    def delete_model(self, model_id):
        """
        Delete a model from the registry
        
        Args:
            model_id (str): ID of the model to delete
            
        Returns:
            bool: True if model was deleted
        """
        if model_id not in self.model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.model_metadata[model_id]
        framework = model_info["framework"]
        
        # Delete model directory
        model_dir = os.path.join(self.models_dir, framework, model_id)
        try:
            shutil.rmtree(model_dir, ignore_errors=True)
        except Exception as e:
            error_msg = f"Error deleting model directory: {str(e)}"
            traceback.print_exc()
            print(error_msg)  # Print but don't fail
        
        # Remove from metadata
        del self.model_metadata[model_id]
        self._save_metadata()
        
        return True
    
    def export_model(self, model_id, export_dir=None, format=None):
        """
        Export a model to a directory
        
        Args:
            model_id (str): ID of the model to export
            export_dir (str): Directory to export to
            format (str): Export format (SavedModel, H5, ONNX)
            
        Returns:
            str: Path to exported model
        """
        if model_id not in self.model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.model_metadata[model_id]
        framework = model_info["framework"]
        
        # Load model
        if framework == "tensorflow":
            model, _ = self.load_model(model_id)
        else:
            raise ValueError(f"Export for {framework} not implemented")
        
        # Create export directory
        if export_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"exported_model_{model_id}_{timestamp}"
        
        os.makedirs(export_dir, exist_ok=True)
        
        # Export model
        try:
            if framework == "tensorflow":
                if format == "h5":
                    export_path = os.path.join(export_dir, "model.h5")
                    model.save(export_path, save_format="h5")
                elif format == "onnx":
                    # Requires tf2onnx
                    try:
                        import tf2onnx
                        export_path = os.path.join(export_dir, "model.onnx")
                        tf2onnx.convert.from_keras(model, output_path=export_path)
                    except ImportError:
                        raise ImportError("tf2onnx is required for ONNX export")
                else:
                    # Default to SavedModel
                    export_path = os.path.join(export_dir, "saved_model")
                    model.save(export_path, save_format="tf")
            else:
                raise ValueError(f"Export for {framework} not implemented")
            
            # Save metadata
            metadata_path = os.path.join(export_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    "id": model_id,
                    "name": model_info["name"],
                    "framework": framework,
                    "version": model_info["version"],
                    "created_at": model_info["created_at"],
                    "exported_at": datetime.now().isoformat(),
                    "metrics": model_info.get("metrics", {})
                }, f, indent=2)
            
            return export_path
        except Exception as e:
            error_msg = f"Error exporting model: {str(e)}"
            traceback.print_exc()
            raise RuntimeError(error_msg)
    
    def import_model(self, model_path, framework, name, description="", tags=None):
        """
        Import a model with existing metadata
        
        Args:
            model_path (str): Path to the model file
            framework (str): Model framework ('tensorflow' or 'pytorch')
            name (str): Model name
            description (str): Model description
            tags (list): List of tags
        
        Returns:
            dict: Model information
        """
        return self.register_model(
            model=model_path,  # For PyTorch models, we just need the path
            framework=framework,
            name=name,
            description=description,
            tags=tags
        )
    
    def _save_metadata(self):
        """Save model metadata to file"""
        metadata_file = os.path.join(self.models_dir, "metadata.json")
        
        with open(metadata_file, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
    
    def _load_metadata(self):
        """Load model metadata from file"""
        metadata_file = os.path.join(self.models_dir, "metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.model_metadata = json.load(f)
    
    def compare_models(self, model_ids):
        """
        Compare multiple models
        
        Args:
            model_ids (list): List of model IDs to compare
            
        Returns:
            dict: Comparison results
        """
        if not model_ids:
            raise ValueError("No model IDs provided for comparison")
        
        models = []
        for model_id in model_ids:
            if model_id not in self.model_metadata:
                raise ValueError(f"Model {model_id} not found")
            
            models.append(self.model_metadata[model_id])
        
        # Collect metrics for comparison
        metrics_comparison = {}
        for model in models:
            model_metrics = model.get("metrics", {})
            for metric_name, metric_value in model_metrics.items():
                if metric_name not in metrics_comparison:
                    metrics_comparison[metric_name] = {}
                
                metrics_comparison[metric_name][model["id"]] = {
                    "value": metric_value,
                    "model_name": model["name"],
                    "model_version": model["version"]
                }
        
        # Find best model for each metric
        best_models = {}
        for metric_name, metric_values in metrics_comparison.items():
            # Determine if higher is better based on metric name
            higher_is_better = not any(term in metric_name.lower() for term in 
                                     ["loss", "error", "mae", "mse", "rmse"])
            
            if higher_is_better:
                best_model_id = max(metric_values.items(), key=lambda x: x[1]["value"])[0]
            else:
                best_model_id = min(metric_values.items(), key=lambda x: x[1]["value"])[0]
            
            best_models[metric_name] = {
                "model_id": best_model_id,
                "model_name": metric_values[best_model_id]["model_name"],
                "value": metric_values[best_model_id]["value"]
            }
        
        return {
            "models": [
                {
                    "id": model["id"],
                    "name": model["name"],
                    "framework": model["framework"],
                    "version": model["version"],
                    "created_at": model["created_at"]
                }
                for model in models
            ],
            "metrics_comparison": metrics_comparison,
            "best_models": best_models
        }
