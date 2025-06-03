# Trajectory Prediction Project

This project implements a deep learning-based trajectory prediction system for vehicles in the environment.

## Project Structure

```
.
├── config/             # Configuration files
├── src/               # Source code
│   ├── models/        # Model definitions
│   ├── datasets/      # Dataset classes
│   ├── utils/         # Utility functions
│   └── registry/      # Registry mechanism
├── scripts/           # Training and evaluation scripts
└── requirements.txt   # Project dependencies
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Training:
```bash
python scripts/train.py --config config/train_config.yaml
```

2. Evaluation:
```bash
python scripts/evaluate.py --config config/eval_config.yaml
```

## Features

- Modular architecture with registry mechanism
- Support for multiple model architectures
- Configurable training and evaluation
- TensorBoard integration for visualization 