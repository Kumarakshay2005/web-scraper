# ---------- Builder Stage ----------
FROM python:3.10-slim AS builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    wget curl unzip gnupg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies into a temporary folder
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- Final Runtime Stage ----------
FROM python:3.10-slim

# Install system dependencies (Chromium + Chromedriver)
RUN apt-get update && apt-get install -y \
    wget curl unzip gnupg \
    chromium chromium-driver \
    fonts-liberation \
    --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for Selenium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Copy Python dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY . .

# Expose port
EXPOSE 8080

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
