FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY cutroom ./cutroom
RUN pip install --no-cache-dir .
ENV CUTROOM_HOME=/data
VOLUME /data
EXPOSE 8000
CMD ["sh", "-c", "python -m cutroom serve --port ${PORT:-8000}"]
