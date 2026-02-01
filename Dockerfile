FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
RUN apt-get update && apt-get upgrade -y && \
    apt-get install libgl1 libgomp1 libglib2.0-0 ffmpeg vim wget curl zip unzip g++ build-essential procps -y && \
    mkdir /app

# Set working directory
WORKDIR /app

# Copy all files from current directory to working directory
COPY . /app

RUN uv sync --frozen
EXPOSE 1995
CMD ["uv", "run", "python", "src/run.py"]