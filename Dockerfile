# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY app.py .
COPY xgboost-model.pkl .

# Expose the necessary ports
EXPOSE 8000 8001

# Command to run the application
CMD ["python", "app.py"]