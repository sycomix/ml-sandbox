"""
Workflow Manager - Handles saving, loading, and managing ML workflows
"""
import os
import json
import time
import uuid
import traceback
from datetime import datetime

class WorkflowManager:
    """
    Manages ML workflows, including saving, loading, and version control
    """
    def __init__(self, storage_dir="workflows"):
        """
        Initialize the workflow manager
        
        Args:
            storage_dir (str): Directory to store workflows
        """
        self.storage_dir = storage_dir
        self.current_workflow = None
        self.workflow_history = []
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
    def create_workflow(self, name, description=""):
        """
        Create a new workflow
        
        Args:
            name (str): Name of the workflow
            description (str): Description of the workflow
            
        Returns:
            dict: New workflow object
        """
        workflow_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        workflow = {
            "id": workflow_id,
            "name": name,
            "description": description,
            "created_at": created_at,
            "updated_at": created_at,
            "version": 1,
            "nodes": {},
            "connections": [],
            "metadata": {
                "author": "ML Sandbox User",
                "framework": "tensorflow",
                "tags": []
            }
        }
        
        self.current_workflow = workflow
        self.workflow_history = [self._create_history_entry("create", workflow)]
        
        return workflow
    
    def save_workflow(self, workflow=None):
        """
        Save a workflow to disk
        
        Args:
            workflow (dict): Workflow to save, defaults to current workflow
            
        Returns:
            dict: Saved workflow with updated metadata
        """
        if workflow is None:
            workflow = self.current_workflow
            
        if workflow is None:
            raise ValueError("No workflow to save")
        
        # Update metadata
        workflow["updated_at"] = datetime.now().isoformat()
        
        # Create filename
        filename = f"{workflow['id']}_v{workflow['version']}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(workflow, f, indent=2)
                
            # Add to history
            self.workflow_history.append(self._create_history_entry("save", workflow))
            
            return workflow
        except Exception as e:
            error_msg = f"Error saving workflow: {str(e)}"
            traceback.print_exc()
            raise RuntimeError(error_msg)
    
    def load_workflow(self, workflow_id, version=None):
        """
        Load a workflow from disk
        
        Args:
            workflow_id (str): ID of the workflow to load
            version (int): Version to load, defaults to latest
            
        Returns:
            dict: Loaded workflow
        """
        # Find workflow files
        workflow_files = []
        for filename in os.listdir(self.storage_dir):
            if filename.startswith(f"{workflow_id}_v") and filename.endswith(".json"):
                workflow_files.append(filename)
        
        if not workflow_files:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Sort by version
        workflow_files.sort(key=lambda f: int(f.split('_v')[1].split('.json')[0]))
        
        # Select version
        if version is not None:
            target_file = f"{workflow_id}_v{version}.json"
            if target_file not in workflow_files:
                raise ValueError(f"Version {version} of workflow {workflow_id} not found")
            selected_file = target_file
        else:
            selected_file = workflow_files[-1]  # Latest version
        
        # Load workflow
        try:
            filepath = os.path.join(self.storage_dir, selected_file)
            with open(filepath, 'r') as f:
                workflow = json.load(f)
                
            self.current_workflow = workflow
            self.workflow_history = [self._create_history_entry("load", workflow)]
            
            return workflow
        except Exception as e:
            error_msg = f"Error loading workflow: {str(e)}"
            traceback.print_exc()
            raise RuntimeError(error_msg)
    
    def list_workflows(self):
        """
        List all available workflows
        
        Returns:
            list: List of workflow metadata
        """
        workflows = {}
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                try:
                    filepath = os.path.join(self.storage_dir, filename)
                    with open(filepath, 'r') as f:
                        workflow = json.load(f)
                        
                    workflow_id = workflow.get("id")
                    if workflow_id:
                        if workflow_id not in workflows:
                            workflows[workflow_id] = {
                                "id": workflow_id,
                                "name": workflow.get("name", "Unnamed"),
                                "description": workflow.get("description", ""),
                                "created_at": workflow.get("created_at", ""),
                                "versions": []
                            }
                        
                        workflows[workflow_id]["versions"].append({
                            "version": workflow.get("version", 1),
                            "updated_at": workflow.get("updated_at", ""),
                            "filename": filename
                        })
                except Exception as e:
                    print(f"Error reading workflow file {filename}: {str(e)}")
        
        # Sort versions
        for workflow_id in workflows:
            workflows[workflow_id]["versions"].sort(key=lambda v: v["version"])
        
        return list(workflows.values())
    
    def update_workflow(self, updates):
        """
        Update the current workflow
        
        Args:
            updates (dict): Updates to apply to the workflow
            
        Returns:
            dict: Updated workflow
        """
        if self.current_workflow is None:
            raise ValueError("No current workflow to update")
        
        # Apply updates
        for key, value in updates.items():
            if key in ["nodes", "connections", "metadata"]:
                if isinstance(value, dict) and isinstance(self.current_workflow[key], dict):
                    self.current_workflow[key].update(value)
                else:
                    self.current_workflow[key] = value
            elif key not in ["id", "created_at"]:
                self.current_workflow[key] = value
        
        # Update metadata
        self.current_workflow["updated_at"] = datetime.now().isoformat()
        
        # Add to history
        self.workflow_history.append(self._create_history_entry("update", self.current_workflow))
        
        return self.current_workflow
    
    def create_version(self):
        """
        Create a new version of the current workflow
        
        Returns:
            dict: New workflow version
        """
        if self.current_workflow is None:
            raise ValueError("No current workflow to version")
        
        # Create new version
        new_version = self.current_workflow.copy()
        new_version["version"] = self.current_workflow["version"] + 1
        new_version["updated_at"] = datetime.now().isoformat()
        
        # Save new version
        self.current_workflow = new_version
        self.workflow_history.append(self._create_history_entry("version", new_version))
        
        return self.save_workflow(new_version)
    
    def export_workflow(self, filepath=None):
        """
        Export the current workflow to a file
        
        Args:
            filepath (str): Path to export to, defaults to timestamped file
            
        Returns:
            str: Path to exported file
        """
        if self.current_workflow is None:
            raise ValueError("No current workflow to export")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workflow_name = self.current_workflow["name"].replace(" ", "_").lower()
            filepath = f"{workflow_name}_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.current_workflow, f, indent=2)
                
            return filepath
        except Exception as e:
            error_msg = f"Error exporting workflow: {str(e)}"
            traceback.print_exc()
            raise RuntimeError(error_msg)
    
    def import_workflow(self, filepath):
        """
        Import a workflow from a file
        
        Args:
            filepath (str): Path to import from
            
        Returns:
            dict: Imported workflow
        """
        try:
            with open(filepath, 'r') as f:
                workflow = json.load(f)
            
            # Validate workflow structure
            required_keys = ["name", "nodes", "connections"]
            for key in required_keys:
                if key not in workflow:
                    raise ValueError(f"Invalid workflow format: missing '{key}'")
            
            # Generate new ID and reset version if not present
            if "id" not in workflow:
                workflow["id"] = str(uuid.uuid4())
            if "version" not in workflow:
                workflow["version"] = 1
            if "created_at" not in workflow:
                workflow["created_at"] = datetime.now().isoformat()
            if "updated_at" not in workflow:
                workflow["updated_at"] = datetime.now().isoformat()
            
            self.current_workflow = workflow
            self.workflow_history = [self._create_history_entry("import", workflow)]
            
            return workflow
        except Exception as e:
            error_msg = f"Error importing workflow: {str(e)}"
            traceback.print_exc()
            raise RuntimeError(error_msg)
    
    def add_node(self, node_data):
        """
        Add a node to the current workflow
        
        Args:
            node_data (dict): Node data
            
        Returns:
            str: ID of the added node
        """
        if self.current_workflow is None:
            raise ValueError("No current workflow to add node to")
        
        node_id = node_data.get("id", str(uuid.uuid4()))
        node_data["id"] = node_id
        
        self.current_workflow["nodes"][node_id] = node_data
        self.workflow_history.append(self._create_history_entry("add_node", node_data))
        
        return node_id
    
    def update_node(self, node_id, updates):
        """
        Update a node in the current workflow
        
        Args:
            node_id (str): ID of the node to update
            updates (dict): Updates to apply
            
        Returns:
            dict: Updated node
        """
        if self.current_workflow is None:
            raise ValueError("No current workflow to update node in")
        
        if node_id not in self.current_workflow["nodes"]:
            raise ValueError(f"Node {node_id} not found in workflow")
        
        node = self.current_workflow["nodes"][node_id]
        
        # Apply updates
        for key, value in updates.items():
            if key != "id":
                node[key] = value
        
        self.workflow_history.append(self._create_history_entry("update_node", node))
        
        return node
    
    def remove_node(self, node_id):
        """
        Remove a node from the current workflow
        
        Args:
            node_id (str): ID of the node to remove
            
        Returns:
            bool: True if node was removed
        """
        if self.current_workflow is None:
            raise ValueError("No current workflow to remove node from")
        
        if node_id not in self.current_workflow["nodes"]:
            raise ValueError(f"Node {node_id} not found in workflow")
        
        # Remove node
        removed_node = self.current_workflow["nodes"].pop(node_id)
        
        # Remove connections involving this node
        self.current_workflow["connections"] = [
            conn for conn in self.current_workflow["connections"]
            if conn["source"] != node_id and conn["target"] != node_id
        ]
        
        self.workflow_history.append(self._create_history_entry("remove_node", removed_node))
        
        return True
    
    def add_connection(self, connection_data):
        """
        Add a connection to the current workflow
        
        Args:
            connection_data (dict): Connection data
            
        Returns:
            dict: Added connection
        """
        if self.current_workflow is None:
            raise ValueError("No current workflow to add connection to")
        
        # Validate connection
        required_keys = ["source", "target"]
        for key in required_keys:
            if key not in connection_data:
                raise ValueError(f"Invalid connection data: missing '{key}'")
        
        # Check if nodes exist
        if connection_data["source"] not in self.current_workflow["nodes"]:
            raise ValueError(f"Source node {connection_data['source']} not found in workflow")
        if connection_data["target"] not in self.current_workflow["nodes"]:
            raise ValueError(f"Target node {connection_data['target']} not found in workflow")
        
        # Add connection
        connection_id = connection_data.get("id", str(uuid.uuid4()))
        connection_data["id"] = connection_id
        
        self.current_workflow["connections"].append(connection_data)
        self.workflow_history.append(self._create_history_entry("add_connection", connection_data))
        
        return connection_data
    
    def remove_connection(self, connection_id):
        """
        Remove a connection from the current workflow
        
        Args:
            connection_id (str): ID of the connection to remove
            
        Returns:
            bool: True if connection was removed
        """
        if self.current_workflow is None:
            raise ValueError("No current workflow to remove connection from")
        
        # Find connection
        connection = None
        for conn in self.current_workflow["connections"]:
            if conn.get("id") == connection_id:
                connection = conn
                break
        
        if connection is None:
            raise ValueError(f"Connection {connection_id} not found in workflow")
        
        # Remove connection
        self.current_workflow["connections"].remove(connection)
        self.workflow_history.append(self._create_history_entry("remove_connection", connection))
        
        return True
    
    def get_workflow_history(self):
        """
        Get the history of the current workflow
        
        Returns:
            list: Workflow history entries
        """
        return self.workflow_history
    
    def _create_history_entry(self, action, data):
        """
        Create a history entry
        
        Args:
            action (str): Action performed
            data (dict): Data associated with the action
            
        Returns:
            dict: History entry
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data
        }
