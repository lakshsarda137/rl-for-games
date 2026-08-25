#!/bin/sh
# Container entrypoint: fetch model weights from S3 (if configured), then serve.
#
#   MODEL_S3_URI   s3://bucket/checkpoints/latest.pt   (optional; skipped if unset
#                  or if the file already exists in data/checkpoints/)
#   PORT           listen port (default 8000)
set -e
cd /app
# Any arguments -> run them instead of the server (diagnostics, one-off scripts).
if [ "$#" -gt 0 ]; then exec "$@"; fi
mkdir -p data/checkpoints data/external_models
if [ -n "$MODEL_S3_URI" ]; then
  dst="data/checkpoints/$(basename "$MODEL_S3_URI")"
  if [ ! -s "$dst" ]; then
    echo "downloading $MODEL_S3_URI -> $dst"
    python -c "import boto3,sys; b,k=sys.argv[1][5:].split('/',1); boto3.client('s3').download_file(b,k,sys.argv[2])" "$MODEL_S3_URI" "$dst"
  fi
fi
ls -la data/checkpoints
exec uvicorn backend:app --app-dir serve --host 0.0.0.0 --port "${PORT:-8000}" --log-level info
