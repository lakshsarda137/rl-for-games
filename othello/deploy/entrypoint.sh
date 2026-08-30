#!/bin/sh
# Container entrypoint: fetch model weights from S3 (if configured), then serve.
#
#   MODEL_S3_URI   s3://bucket/checkpoints/latest.pt[,s3://.../early.pt]   (optional;
#                  comma-separated; each skipped if unset or already present)
#   PORT           listen port (default 8000)
set -e
cd /app
# Any arguments -> run them instead of the server (diagnostics, one-off scripts).
if [ "$#" -gt 0 ]; then exec "$@"; fi
mkdir -p data/checkpoints data/external_models
for uri in $(echo "$MODEL_S3_URI" | tr ',' ' '); do
  dst="data/checkpoints/$(basename "$uri")"
  if [ ! -s "$dst" ]; then
    echo "downloading $uri -> $dst"
    python -c "import boto3,sys; b,k=sys.argv[1][5:].split('/',1); boto3.client('s3').download_file(b,k,sys.argv[2])" "$uri" "$dst"
  fi
done
ls -la data/checkpoints
exec uvicorn backend:app --app-dir serve --host 0.0.0.0 --port "${PORT:-8000}" --log-level info
