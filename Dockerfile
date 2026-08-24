FROM python:3.11-slim
WORKDIR /app

# Install system deps needed by some science packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

ENV PORT=8080
EXPOSE 8080

# Run Streamlit on the platform-provided port
CMD ["sh", "-c", "streamlit run dashboard.py --server.port ${PORT} --server.address 0.0.0.0 --server.enableCORS false --server.headless true"]
