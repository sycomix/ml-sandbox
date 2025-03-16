# ML Sandbox: Interactive AI/ML Development Environment

An interactive AI/ML development environment inspired by Node-RED, enabling users to assemble and test machine learning experiments in real-time using draggable blocks of neural network code.

## Features

- Drag-and-drop interface for selecting and arranging ML components
- Core ML building blocks including neural network layers, activations, optimizers, and loss functions
- Real-time execution of assembled workflows
- Support for end-to-end ML pipelines from data ingestion to model deployment
- Interactive visualization of component connections and runtime statistics
- Error handling and debugging capabilities
- Scalability across different ML block combinations
- Integration with popular ML frameworks (TensorFlow, PyTorch)
- Version control and collaboration features
- Security measures for data privacy and code integrity

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-sandbox.git
cd ml-sandbox
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows:
   ```bash
   venv\Scripts\activate
   ```
   - macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Start building your ML pipeline by dragging and dropping components from the sidebar onto the canvas.

## Project Structure

- `app.py`: Main application file
- `static/`: Static files (CSS, JavaScript, images)
- `templates/`: HTML templates
- `models/`: ML model definitions
- `utils/`: Utility functions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
