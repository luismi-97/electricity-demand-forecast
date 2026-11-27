# Prediction API. Build:  docker build -t edf-api .          (LightGBM only, small image)
#                         docker build -t edf-api --build-arg WITH_NN=1 .   (adds TensorFlow for the blend)
# Run:    docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models edf-api
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

ARG WITH_NN=0
ENV EDF_HOME=/app PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[api]" && if [ "$WITH_NN" = "1" ]; then pip install ".[nn]"; fi

RUN useradd --create-home appuser && mkdir -p /app/data /app/models && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "edf.api:app", "--host", "0.0.0.0", "--port", "8000"]
