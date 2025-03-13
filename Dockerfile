# Use a base image with Python and CUDA support for GPU training
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh

# Default command: Run fine-tuning
CMD ["bash", "scripts/run_finetuning.sh"]