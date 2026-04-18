FROM python:3.11-slim

WORKDIR /app

# system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

#copying dependency files first for better layer caching
COPY requirements.txt .
COPY pyproject.toml ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copying the repo
COPY . .

# default command just shows evaluator help
# the actual milestone commands are passed with docker run
CMD ["python", "-m", "scripts.evaluator", "--help"]