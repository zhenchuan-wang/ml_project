# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copy the project files
COPY pyproject.toml poetry.lock ./

# Install project dependencies (without creating a virtual environment)
RUN poetry config virtualenvs.create false && poetry install --no-root

# Copy the rest of the application code
COPY . .


# Command to run application
CMD ["python", "src/task1.py"]