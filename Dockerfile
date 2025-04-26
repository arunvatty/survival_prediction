FROM python:3.9-slim-buster

WORKDIR /app

# Install gcc for compiling dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies with optimizations
RUN pip install --no-cache-dir --compile -r requirements.txt && \
    # Remove pip cache
    rm -rf /root/.cache/pip/*

# Copy model and application files
COPY xgboost-model.pkl .
COPY app.py .

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application
CMD ["python", "app.py"]