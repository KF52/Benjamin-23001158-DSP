FROM python:3.10-slim
RUN apt-get update && apt-get install -y curl

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the application code and model files
COPY *.py /app/
COPY *.pkl /app/

# Expose the port
EXPOSE 8088

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088"]