"""
Error Handler - Provides robust error handling for ML Sandbox
"""
import sys
import traceback
import logging
import json
import os
from datetime import datetime

class ErrorHandler:
    """
    Handles errors and exceptions in the ML Sandbox application
    """
    def __init__(self, log_dir="logs", log_level=logging.INFO):
        """
        Initialize the error handler
        
        Args:
            log_dir (str): Directory to store log files
            log_level (int): Logging level
        """
        self.log_dir = log_dir
        self.log_level = log_level
        self.errors = []
        self.warnings = []
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure logging
        self._configure_logging()
        
    def _configure_logging(self):
        """Configure the logging system"""
        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"ml_sandbox_{timestamp}.log")
        
        # Configure root logger
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create logger for this class
        self.logger = logging.getLogger('ml_sandbox.error_handler')
        self.logger.info("Error handler initialized")
        
    def handle_exception(self, exception, component=None, context=None):
        """
        Handle an exception
        
        Args:
            exception (Exception): The exception to handle
            component (str): The component where the exception occurred
            context (dict): Additional context information
            
        Returns:
            dict: Error information
        """
        error_type = type(exception).__name__
        error_message = str(exception)
        error_traceback = traceback.format_exc()
        
        # Create error record
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_message,
            "traceback": error_traceback,
            "component": component,
            "context": context
        }
        
        # Add to errors list
        self.errors.append(error_info)
        
        # Log the error
        self.logger.error(
            f"Exception in {component or 'unknown'}: {error_type}: {error_message}",
            exc_info=True
        )
        
        if context:
            self.logger.error(f"Context: {json.dumps(context, default=str)}")
        
        return error_info
    
    def handle_workflow_error(self, workflow_id, node_id, exception, context=None):
        """
        Handle a workflow execution error
        
        Args:
            workflow_id (str): ID of the workflow
            node_id (str): ID of the node where the error occurred
            exception (Exception): The exception to handle
            context (dict): Additional context information
            
        Returns:
            dict: Error information
        """
        error_context = {
            "workflow_id": workflow_id,
            "node_id": node_id,
            **(context or {})
        }
        
        return self.handle_exception(
            exception,
            component=f"workflow.node.{node_id}",
            context=error_context
        )
    
    def add_warning(self, message, component=None, context=None):
        """
        Add a warning
        
        Args:
            message (str): Warning message
            component (str): The component where the warning occurred
            context (dict): Additional context information
            
        Returns:
            dict: Warning information
        """
        warning_info = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "component": component,
            "context": context
        }
        
        # Add to warnings list
        self.warnings.append(warning_info)
        
        # Log the warning
        self.logger.warning(
            f"Warning in {component or 'unknown'}: {message}"
        )
        
        if context:
            self.logger.warning(f"Context: {json.dumps(context, default=str)}")
        
        return warning_info
    
    def get_errors(self, limit=None, component=None):
        """
        Get recent errors
        
        Args:
            limit (int): Maximum number of errors to return
            component (str): Filter errors by component
            
        Returns:
            list: List of error information
        """
        filtered_errors = self.errors
        
        if component:
            filtered_errors = [e for e in filtered_errors if e.get("component") == component]
        
        if limit:
            filtered_errors = filtered_errors[-limit:]
            
        return filtered_errors
    
    def get_warnings(self, limit=None, component=None):
        """
        Get recent warnings
        
        Args:
            limit (int): Maximum number of warnings to return
            component (str): Filter warnings by component
            
        Returns:
            list: List of warning information
        """
        filtered_warnings = self.warnings
        
        if component:
            filtered_warnings = [w for w in filtered_warnings if w.get("component") == component]
        
        if limit:
            filtered_warnings = filtered_warnings[-limit:]
            
        return filtered_warnings
    
    def clear_errors(self):
        """Clear all errors"""
        self.errors = []
        
    def clear_warnings(self):
        """Clear all warnings"""
        self.warnings = []
        
    def export_logs(self, filepath=None):
        """
        Export logs to a file
        
        Args:
            filepath (str): Path to export to, defaults to timestamped file
            
        Returns:
            str: Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.log_dir, f"ml_sandbox_logs_{timestamp}.json")
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
                
            self.logger.info(f"Logs exported to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error exporting logs: {str(e)}", exc_info=True)
            raise
    
    def validate_workflow(self, workflow):
        """
        Validate a workflow and report any issues
        
        Args:
            workflow (dict): Workflow to validate
            
        Returns:
            dict: Validation results
        """
        issues = []
        
        # Check required fields
        required_fields = ["id", "name", "nodes", "connections"]
        for field in required_fields:
            if field not in workflow:
                issues.append({
                    "type": "error",
                    "message": f"Missing required field: {field}"
                })
        
        # Check nodes
        if "nodes" in workflow:
            nodes = workflow["nodes"]
            if not isinstance(nodes, dict):
                issues.append({
                    "type": "error",
                    "message": "Nodes must be an object"
                })
            else:
                # Check each node
                for node_id, node in nodes.items():
                    if "type" not in node:
                        issues.append({
                            "type": "error",
                            "message": f"Node {node_id} is missing type"
                        })
                    if "category" not in node:
                        issues.append({
                            "type": "error",
                            "message": f"Node {node_id} is missing category"
                        })
        
        # Check connections
        if "connections" in workflow:
            connections = workflow["connections"]
            if not isinstance(connections, list):
                issues.append({
                    "type": "error",
                    "message": "Connections must be an array"
                })
            else:
                # Check each connection
                for i, connection in enumerate(connections):
                    if "source" not in connection:
                        issues.append({
                            "type": "error",
                            "message": f"Connection {i} is missing source"
                        })
                    elif "nodes" in workflow and connection["source"] not in workflow["nodes"]:
                        issues.append({
                            "type": "error",
                            "message": f"Connection {i} references non-existent source node: {connection['source']}"
                        })
                    
                    if "target" not in connection:
                        issues.append({
                            "type": "error",
                            "message": f"Connection {i} is missing target"
                        })
                    elif "nodes" in workflow and connection["target"] not in workflow["nodes"]:
                        issues.append({
                            "type": "error",
                            "message": f"Connection {i} references non-existent target node: {connection['target']}"
                        })
        
        # Check for cycles
        if "nodes" in workflow and "connections" in workflow:
            try:
                self._check_for_cycles(workflow)
            except Exception as e:
                issues.append({
                    "type": "error",
                    "message": f"Workflow contains cycles: {str(e)}"
                })
        
        # Log validation results
        if issues:
            for issue in issues:
                if issue["type"] == "error":
                    self.logger.error(f"Workflow validation error: {issue['message']}")
                else:
                    self.logger.warning(f"Workflow validation warning: {issue['message']}")
        else:
            self.logger.info("Workflow validation successful")
        
        return {
            "valid": len([i for i in issues if i["type"] == "error"]) == 0,
            "issues": issues
        }
    
    def _check_for_cycles(self, workflow):
        """
        Check if a workflow contains cycles
        
        Args:
            workflow (dict): Workflow to check
            
        Raises:
            ValueError: If cycles are detected
        """
        nodes = workflow["nodes"]
        connections = workflow["connections"]
        
        # Build adjacency list
        graph = {node_id: [] for node_id in nodes}
        for conn in connections:
            source = conn["source"]
            target = conn["target"]
            graph[source].append(target)
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in graph.get(node_id, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    raise ValueError(f"Cycle detected involving nodes {node_id} and {neighbor}")
            
            rec_stack.remove(node_id)
            return False
        
        # Check each node
        for node_id in nodes:
            if node_id not in visited:
                dfs(node_id)
    
    def monitor_execution(self, execution_id, status, stats=None):
        """
        Monitor workflow execution
        
        Args:
            execution_id (str): ID of the execution
            status (str): Execution status
            stats (dict): Execution statistics
            
        Returns:
            dict: Monitoring information
        """
        monitoring_info = {
            "timestamp": datetime.now().isoformat(),
            "execution_id": execution_id,
            "status": status,
            "stats": stats
        }
        
        # Log monitoring information
        if status == "started":
            self.logger.info(f"Execution {execution_id} started")
        elif status == "completed":
            self.logger.info(f"Execution {execution_id} completed")
            if stats:
                self.logger.info(f"Execution stats: {json.dumps(stats, default=str)}")
        elif status == "error":
            self.logger.error(f"Execution {execution_id} failed")
            if stats:
                self.logger.error(f"Execution stats: {json.dumps(stats, default=str)}")
        else:
            self.logger.info(f"Execution {execution_id} status: {status}")
        
        return monitoring_info
