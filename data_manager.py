"""
Data Manager - Handles dataset operations for ML Sandbox
"""
import os
import json
import numpy as np
import pandas as pd
import pickle
import uuid
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder

class DataManager:
    """
    Manages datasets for ML workflows, including loading, preprocessing, and splitting
    """
    def __init__(self, data_dir="datasets"):
        """
        Initialize the data manager
        
        Args:
            data_dir (str): Directory to store datasets
        """
        self.data_dir = data_dir
        self.current_dataset = None
        self.preprocessors = {}
        self.dataset_metadata = {}
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "splits"), exist_ok=True)
        
    def load_dataset(self, file_path, dataset_name=None, file_type=None):
        """
        Load a dataset from a file
        
        Args:
            file_path (str): Path to the dataset file
            dataset_name (str): Name for the dataset, defaults to filename
            file_type (str): Type of file (csv, json, excel), defaults to auto-detect
            
        Returns:
            dict: Dataset information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Determine file type if not provided
        if file_type is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                file_type = 'csv'
            elif ext == '.json':
                file_type = 'json'
            elif ext in ['.xls', '.xlsx']:
                file_type = 'excel'
            elif ext in ['.npy', '.npz']:
                file_type = 'numpy'
            elif ext == '.pkl':
                file_type = 'pickle'
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        
        # Load data based on file type
        try:
            if file_type == 'csv':
                data = pd.read_csv(file_path)
            elif file_type == 'json':
                data = pd.read_json(file_path)
            elif file_type == 'excel':
                data = pd.read_excel(file_path)
            elif file_type == 'numpy':
                data = np.load(file_path, allow_pickle=True)
                # Convert numpy array to DataFrame if possible
                if isinstance(data, np.ndarray):
                    data = pd.DataFrame(data)
            elif file_type == 'pickle':
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                # Convert to DataFrame if not already
                if not isinstance(data, pd.DataFrame):
                    if isinstance(data, dict):
                        data = pd.DataFrame.from_dict(data)
                    elif isinstance(data, np.ndarray):
                        data = pd.DataFrame(data)
                    else:
                        raise ValueError("Pickle file must contain DataFrame-compatible data")
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            error_msg = f"Error loading dataset: {str(e)}"
            traceback.print_exc()
            raise RuntimeError(error_msg)
        
        # Generate dataset ID and name if not provided
        dataset_id = str(uuid.uuid4())
        if dataset_name is None:
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create dataset metadata
        self.current_dataset = {
            "id": dataset_id,
            "name": dataset_name,
            "original_file": file_path,
            "file_type": file_type,
            "created_at": datetime.now().isoformat(),
            "rows": len(data),
            "columns": len(data.columns) if hasattr(data, 'columns') else 0,
            "column_types": {col: str(data[col].dtype) for col in data.columns} if hasattr(data, 'columns') else {},
            "has_missing_values": data.isnull().any().any() if hasattr(data, 'isnull') else False
        }
        
        # Save raw data
        raw_file_path = os.path.join(self.data_dir, "raw", f"{dataset_id}.pkl")
        with open(raw_file_path, 'wb') as f:
            pickle.dump(data, f)
        
        self.current_dataset["raw_file"] = raw_file_path
        self.dataset_metadata[dataset_id] = self.current_dataset
        
        # Save metadata
        self._save_metadata()
        
        return {
            "dataset_id": dataset_id,
            "name": dataset_name,
            "shape": data.shape if hasattr(data, 'shape') else None,
            "columns": list(data.columns) if hasattr(data, 'columns') else None,
            "preview": data.head(5).to_dict('records') if hasattr(data, 'head') else None
        }
    
    def get_dataset_info(self, dataset_id=None):
        """
        Get information about a dataset
        
        Args:
            dataset_id (str): ID of the dataset, defaults to current dataset
            
        Returns:
            dict: Dataset information
        """
        if dataset_id is None:
            if self.current_dataset is None:
                raise ValueError("No current dataset")
            dataset_id = self.current_dataset["id"]
        
        if dataset_id not in self.dataset_metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset_info = self.dataset_metadata[dataset_id]
        
        # Load raw data to get preview
        raw_file = dataset_info["raw_file"]
        with open(raw_file, 'rb') as f:
            data = pickle.load(f)
        
        return {
            "dataset_id": dataset_id,
            "name": dataset_info["name"],
            "shape": data.shape if hasattr(data, 'shape') else None,
            "columns": list(data.columns) if hasattr(data, 'columns') else None,
            "column_types": dataset_info["column_types"],
            "has_missing_values": dataset_info["has_missing_values"],
            "preview": data.head(5).to_dict('records') if hasattr(data, 'head') else None
        }
    
    def list_datasets(self):
        """
        List all available datasets
        
        Returns:
            list: List of dataset metadata
        """
        return [
            {
                "id": dataset_id,
                "name": info["name"],
                "created_at": info["created_at"],
                "rows": info["rows"],
                "columns": info["columns"]
            }
            for dataset_id, info in self.dataset_metadata.items()
        ]
    
    def preprocess_dataset(self, dataset_id=None, operations=None):
        """
        Preprocess a dataset
        
        Args:
            dataset_id (str): ID of the dataset, defaults to current dataset
            operations (list): List of preprocessing operations
            
        Returns:
            dict: Preprocessing results
        """
        if dataset_id is None:
            if self.current_dataset is None:
                raise ValueError("No current dataset")
            dataset_id = self.current_dataset["id"]
        
        if dataset_id not in self.dataset_metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load raw data
        raw_file = self.dataset_metadata[dataset_id]["raw_file"]
        with open(raw_file, 'rb') as f:
            data = pickle.load(f)
        
        # Initialize preprocessors for this dataset if not exists
        if dataset_id not in self.preprocessors:
            self.preprocessors[dataset_id] = {}
        
        # Apply preprocessing operations
        if operations:
            for op in operations:
                op_type = op.get("type")
                columns = op.get("columns", [])
                
                if not columns:
                    # If no columns specified, use all numeric or all columns depending on operation
                    if op_type in ["standardize", "normalize", "log_transform"]:
                        columns = data.select_dtypes(include=np.number).columns.tolist()
                    else:
                        columns = data.columns.tolist()
                
                # Apply operation
                if op_type == "standardize":
                    scaler = StandardScaler()
                    data[columns] = scaler.fit_transform(data[columns])
                    self.preprocessors[dataset_id]["standardize"] = {
                        "scaler": scaler,
                        "columns": columns
                    }
                
                elif op_type == "normalize":
                    scaler = MinMaxScaler()
                    data[columns] = scaler.fit_transform(data[columns])
                    self.preprocessors[dataset_id]["normalize"] = {
                        "scaler": scaler,
                        "columns": columns
                    }
                
                elif op_type == "one_hot_encode":
                    for col in columns:
                        if col in data.columns and data[col].dtype == 'object':
                            encoder = OneHotEncoder(sparse=False, drop='first')
                            encoded = encoder.fit_transform(data[[col]])
                            encoded_df = pd.DataFrame(
                                encoded,
                                columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                            )
                            data = pd.concat([data.drop(col, axis=1), encoded_df], axis=1)
                            
                            if "one_hot_encode" not in self.preprocessors[dataset_id]:
                                self.preprocessors[dataset_id]["one_hot_encode"] = {}
                            
                            self.preprocessors[dataset_id]["one_hot_encode"][col] = {
                                "encoder": encoder,
                                "categories": encoder.categories_[0].tolist()
                            }
                
                elif op_type == "label_encode":
                    for col in columns:
                        if col in data.columns and data[col].dtype == 'object':
                            encoder = LabelEncoder()
                            data[col] = encoder.fit_transform(data[col])
                            
                            if "label_encode" not in self.preprocessors[dataset_id]:
                                self.preprocessors[dataset_id]["label_encode"] = {}
                            
                            self.preprocessors[dataset_id]["label_encode"][col] = {
                                "encoder": encoder,
                                "classes": encoder.classes_.tolist()
                            }
                
                elif op_type == "fill_missing":
                    strategy = op.get("strategy", "mean")
                    for col in columns:
                        if col in data.columns:
                            if strategy == "mean" and np.issubdtype(data[col].dtype, np.number):
                                fill_value = data[col].mean()
                            elif strategy == "median" and np.issubdtype(data[col].dtype, np.number):
                                fill_value = data[col].median()
                            elif strategy == "mode":
                                fill_value = data[col].mode()[0]
                            elif strategy == "constant":
                                fill_value = op.get("value", 0)
                            else:
                                continue
                            
                            data[col] = data[col].fillna(fill_value)
                            
                            if "fill_missing" not in self.preprocessors[dataset_id]:
                                self.preprocessors[dataset_id]["fill_missing"] = {}
                            
                            self.preprocessors[dataset_id]["fill_missing"][col] = {
                                "strategy": strategy,
                                "value": fill_value
                            }
                
                elif op_type == "drop_columns":
                    data = data.drop(columns, axis=1)
                    
                    if "drop_columns" not in self.preprocessors[dataset_id]:
                        self.preprocessors[dataset_id]["drop_columns"] = []
                    
                    self.preprocessors[dataset_id]["drop_columns"].extend(columns)
                
                elif op_type == "drop_missing":
                    threshold = op.get("threshold", None)
                    if threshold:
                        # Drop rows with more than threshold% missing values
                        data = data.dropna(thresh=int((1 - threshold) * len(data.columns)))
                    else:
                        # Drop rows with any missing values in specified columns
                        data = data.dropna(subset=columns)
                
                elif op_type == "log_transform":
                    for col in columns:
                        if col in data.columns and np.issubdtype(data[col].dtype, np.number):
                            # Add small constant to avoid log(0)
                            min_val = data[col].min()
                            shift = 0
                            if min_val <= 0:
                                shift = abs(min_val) + 1
                            
                            data[col] = np.log(data[col] + shift)
                            
                            if "log_transform" not in self.preprocessors[dataset_id]:
                                self.preprocessors[dataset_id]["log_transform"] = {}
                            
                            self.preprocessors[dataset_id]["log_transform"][col] = {
                                "shift": shift
                            }
        
        # Save processed data
        processed_file_path = os.path.join(self.data_dir, "processed", f"{dataset_id}.pkl")
        with open(processed_file_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Update dataset metadata
        self.dataset_metadata[dataset_id]["processed_file"] = processed_file_path
        self.dataset_metadata[dataset_id]["processed_at"] = datetime.now().isoformat()
        self.dataset_metadata[dataset_id]["processed_rows"] = len(data)
        self.dataset_metadata[dataset_id]["processed_columns"] = len(data.columns)
        
        # Save metadata
        self._save_metadata()
        
        return {
            "dataset_id": dataset_id,
            "name": self.dataset_metadata[dataset_id]["name"],
            "shape": data.shape,
            "columns": list(data.columns),
            "preview": data.head(5).to_dict('records')
        }
    
    def split_dataset(self, dataset_id=None, target_column=None, test_size=0.2, val_size=0.1, 
                     random_state=42, stratify=False):
        """
        Split a dataset into train, validation, and test sets
        
        Args:
            dataset_id (str): ID of the dataset, defaults to current dataset
            target_column (str): Name of the target column
            test_size (float): Proportion of data for test set
            val_size (float): Proportion of data for validation set
            random_state (int): Random seed for reproducibility
            stratify (bool): Whether to stratify splits based on target column
            
        Returns:
            dict: Split information
        """
        if dataset_id is None:
            if self.current_dataset is None:
                raise ValueError("No current dataset")
            dataset_id = self.current_dataset["id"]
        
        if dataset_id not in self.dataset_metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load processed data if available, otherwise raw data
        if "processed_file" in self.dataset_metadata[dataset_id]:
            data_file = self.dataset_metadata[dataset_id]["processed_file"]
        else:
            data_file = self.dataset_metadata[dataset_id]["raw_file"]
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        # Validate target column
        if target_column and target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Split features and target if target column provided
        if target_column:
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            
            # Determine stratify parameter
            stratify_param = y if stratify else None
            
            # First split: train+val and test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
            )
            
            # Second split: train and val
            # Adjust val_size to be relative to train_val size
            val_size_adjusted = val_size / (1 - test_size)
            
            # Determine stratify parameter for second split
            stratify_param_2 = y_train_val if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size_adjusted, 
                random_state=random_state, stratify=stratify_param_2
            )
            
            # Recombine features and target for saving
            train_data = pd.concat([X_train, y_train], axis=1)
            val_data = pd.concat([X_val, y_val], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
        else:
            # Split without separate target
            train_val_data, test_data = train_test_split(
                data, test_size=test_size, random_state=random_state
            )
            
            # Adjust val_size to be relative to train_val size
            val_size_adjusted = val_size / (1 - test_size)
            
            train_data, val_data = train_test_split(
                train_val_data, test_size=val_size_adjusted, random_state=random_state
            )
        
        # Save split datasets
        split_dir = os.path.join(self.data_dir, "splits", dataset_id)
        os.makedirs(split_dir, exist_ok=True)
        
        train_file = os.path.join(split_dir, "train.pkl")
        val_file = os.path.join(split_dir, "val.pkl")
        test_file = os.path.join(split_dir, "test.pkl")
        
        with open(train_file, 'wb') as f:
            pickle.dump(train_data, f)
        
        with open(val_file, 'wb') as f:
            pickle.dump(val_data, f)
        
        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        # Update dataset metadata
        self.dataset_metadata[dataset_id]["split_info"] = {
            "target_column": target_column,
            "test_size": test_size,
            "val_size": val_size,
            "random_state": random_state,
            "stratify": stratify,
            "train_file": train_file,
            "val_file": val_file,
            "test_file": test_file,
            "train_shape": train_data.shape,
            "val_shape": val_data.shape,
            "test_shape": test_data.shape,
            "split_at": datetime.now().isoformat()
        }
        
        # Save metadata
        self._save_metadata()
        
        return {
            "dataset_id": dataset_id,
            "name": self.dataset_metadata[dataset_id]["name"],
            "target_column": target_column,
            "train_shape": train_data.shape,
            "val_shape": val_data.shape,
            "test_shape": test_data.shape,
            "train_preview": train_data.head(5).to_dict('records'),
            "val_preview": val_data.head(5).to_dict('records'),
            "test_preview": test_data.head(5).to_dict('records')
        }
    
    def get_split_data(self, dataset_id=None, split="train"):
        """
        Get split dataset
        
        Args:
            dataset_id (str): ID of the dataset, defaults to current dataset
            split (str): Split to get (train, val, test)
            
        Returns:
            pandas.DataFrame: Split data
        """
        if dataset_id is None:
            if self.current_dataset is None:
                raise ValueError("No current dataset")
            dataset_id = self.current_dataset["id"]
        
        if dataset_id not in self.dataset_metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        if "split_info" not in self.dataset_metadata[dataset_id]:
            raise ValueError(f"Dataset {dataset_id} has not been split")
        
        split_info = self.dataset_metadata[dataset_id]["split_info"]
        
        if split == "train":
            file_path = split_info["train_file"]
        elif split == "val":
            file_path = split_info["val_file"]
        elif split == "test":
            file_path = split_info["test_file"]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def get_features_targets(self, dataset_id=None, split="train"):
        """
        Get features and targets from a split dataset
        
        Args:
            dataset_id (str): ID of the dataset, defaults to current dataset
            split (str): Split to get (train, val, test)
            
        Returns:
            tuple: (X, y) features and targets
        """
        if dataset_id is None:
            if self.current_dataset is None:
                raise ValueError("No current dataset")
            dataset_id = self.current_dataset["id"]
        
        if dataset_id not in self.dataset_metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        if "split_info" not in self.dataset_metadata[dataset_id]:
            raise ValueError(f"Dataset {dataset_id} has not been split")
        
        split_info = self.dataset_metadata[dataset_id]["split_info"]
        target_column = split_info.get("target_column")
        
        if not target_column:
            raise ValueError(f"No target column specified for dataset {dataset_id}")
        
        data = self.get_split_data(dataset_id, split)
        
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        return X, y
    
    def export_dataset(self, dataset_id=None, format="csv", processed=True):
        """
        Export a dataset to a file
        
        Args:
            dataset_id (str): ID of the dataset, defaults to current dataset
            format (str): Export format (csv, json, pickle)
            processed (bool): Whether to export processed data
            
        Returns:
            str: Path to exported file
        """
        if dataset_id is None:
            if self.current_dataset is None:
                raise ValueError("No current dataset")
            dataset_id = self.current_dataset["id"]
        
        if dataset_id not in self.dataset_metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Determine which data to export
        if processed and "processed_file" in self.dataset_metadata[dataset_id]:
            data_file = self.dataset_metadata[dataset_id]["processed_file"]
        else:
            data_file = self.dataset_metadata[dataset_id]["raw_file"]
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        # Create export filename
        dataset_name = self.dataset_metadata[dataset_id]["name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"{dataset_name}_{timestamp}.{format}"
        
        # Export data
        try:
            if format == "csv":
                data.to_csv(export_filename, index=False)
            elif format == "json":
                data.to_json(export_filename, orient="records")
            elif format == "pickle":
                with open(export_filename, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            return export_filename
        except Exception as e:
            error_msg = f"Error exporting dataset: {str(e)}"
            traceback.print_exc()
            raise RuntimeError(error_msg)
    
    def import_dataset(self, file_path, dataset_name=None, metadata=None):
        """
        Import a dataset with existing metadata
        
        Args:
            file_path (str): Path to the dataset file
            dataset_name (str): Name for the dataset
            metadata (dict): Existing metadata for the dataset
        
        Returns:
            dict: Dataset information
        """
        # Load the dataset
        dataset_info = self.load_dataset(file_path, dataset_name)
        
        if metadata:
            # Update with existing metadata while preserving new file paths
            metadata.update({
                'raw_file': dataset_info['raw_file'],
                'processed_file': dataset_info.get('processed_file'),
                'split_files': dataset_info.get('split_files')
            })
            self.dataset_metadata[dataset_info['dataset_id']] = metadata
            self._save_metadata()
        
        return dataset_info
    
    def _save_metadata(self):
        """Save dataset metadata to file"""
        metadata_file = os.path.join(self.data_dir, "metadata.json")
        
        # Convert metadata to serializable format
        serializable_metadata = {}
        for dataset_id, metadata in self.dataset_metadata.items():
            serializable_metadata[dataset_id] = {
                k: v for k, v in metadata.items() 
                if k not in ["data", "X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
            }
        
        with open(metadata_file, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
    
    def _load_metadata(self):
        """Load dataset metadata from file"""
        metadata_file = os.path.join(self.data_dir, "metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.dataset_metadata = json.load(f)
    
    def generate_data_summary(self, dataset_id=None):
        """
        Generate a summary of a dataset
        
        Args:
            dataset_id (str): ID of the dataset, defaults to current dataset
            
        Returns:
            dict: Dataset summary
        """
        if dataset_id is None:
            if self.current_dataset is None:
                raise ValueError("No current dataset")
            dataset_id = self.current_dataset["id"]
        
        if dataset_id not in self.dataset_metadata:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        # Load processed data if available, otherwise raw data
        if "processed_file" in self.dataset_metadata[dataset_id]:
            data_file = self.dataset_metadata[dataset_id]["processed_file"]
        else:
            data_file = self.dataset_metadata[dataset_id]["raw_file"]
        
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        # Generate summary statistics
        summary = {
            "dataset_id": dataset_id,
            "name": self.dataset_metadata[dataset_id]["name"],
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(data[col].dtype) for col in data.columns},
            "missing_values": {col: int(data[col].isnull().sum()) for col in data.columns},
            "numeric_stats": {},
            "categorical_stats": {}
        }
        
        # Numeric column statistics
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            summary["numeric_stats"][col] = {
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "mean": float(data[col].mean()),
                "median": float(data[col].median()),
                "std": float(data[col].std())
            }
        
        # Categorical column statistics
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = data[col].value_counts().to_dict()
            # Limit to top 10 categories
            if len(value_counts) > 10:
                top_categories = {k: value_counts[k] for k in list(value_counts.keys())[:10]}
                top_categories["other"] = sum(value_counts[k] for k in list(value_counts.keys())[10:])
                value_counts = top_categories
            
            summary["categorical_stats"][col] = {
                "unique_values": int(data[col].nunique()),
                "value_counts": value_counts
            }
        
        return summary
