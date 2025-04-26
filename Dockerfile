FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app.py .
COPY xgboost-model.pkl .

# Expose the port the app runs on
EXPOSE 8001

# Command to run the application
CMD ["python", "app.py"]