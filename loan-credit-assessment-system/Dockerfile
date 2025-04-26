FROM python:3.10-slim
# Set the environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install the dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
  pip install -r requirements.txt

# Copy the project
COPY . /app/

# Copy the entrypoint script to the image
COPY entrypoint.sh /app/
# Make the entrypoint script executable
RUN chmod +x ./entrypoint.sh

# Set the port
EXPOSE 8000

# Entrypoint script
ENTRYPOINT ["./entrypoint.sh"]

