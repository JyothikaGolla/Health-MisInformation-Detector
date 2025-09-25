
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for spaCy
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy all source code
COPY . .

# Ensure saved_models and dataset.csv are available
RUN mkdir -p /app/saved_models

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]