# Use official Python lightweight runtime as parent image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for OpenCV, PIL, and general libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies manifests
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files to workdir
COPY . /app/

# Set environment variables for configurations
ENV HOST=0.0.0.0
ENV PORT=7860
ENV TF_USE_LEGACY_KERAS=1

# Expose port
EXPOSE 7860

# Run Streamlit startup command
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
