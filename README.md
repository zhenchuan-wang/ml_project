# Multi-Task Learning with Sentence Transformers

## Overview
This project implements a multi-task learning framework using sentence transformers. It includes:
- Task 1: Sentence Transformer Implementation
- Task 2: Multi-Task Learning Expansion
- Task 3: Training Considerations
- Task 4: Training Loop Implementation

## How to Run
### Without Docker
1. Install dependencies using Poetry:
   ```bash
   poetry install

2. Run each task:
    ```bash 
    poetry run python src/task1.py
    poetry run python src/task2.py
    poetry run python src/task4.py

### With Docker
1. Build the Docker image
    ```bash
    docker build -t ml_project .
2. Run the Docker container:
    ```bash
    docker run ml_project
3. Run a Specific Task:
    ```bash
    docker run my-ml-project python src/task2.py
