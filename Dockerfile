# Use official Python image
FROM python:3.10-slim

# Install system dependencies (Chromium + Chromedriver)
RUN apt-get update && apt-get install -y \
    wget curl unzip gnupg \
    chromium chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Selenium
ENV CHROME_usr=/usr/bin/chromium
ENV CHROME_Driver=/usr/bin/chromedriver

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 8080

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
