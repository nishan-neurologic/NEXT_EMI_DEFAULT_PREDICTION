# Use the official Python image as the base image
FROM python:3.9.15
FROM tensorflow/tensorflow:2.8.1
# FROM arm64v8/python:3.8-slim
#gFROM nvidia/cuda:11.2.2-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
&& rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt /app
RUN pip install --upgrade pip
# Install required packages
RUN pip install -r requirements.txt


# Copy the rest of the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 5000

WORKDIR /app/src

# Start the application
CMD ["python", "app.py"]

