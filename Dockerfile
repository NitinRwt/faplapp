FROM python:3.10

WORKDIR /app

# Install required dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application (Fix: Bind to 0.0.0.0)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
