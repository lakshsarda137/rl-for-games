# Othello AlphaZero play server — FastAPI + CPU torch + Edax, one container.
# Build from the othello/ directory:   docker build -f deploy/Dockerfile -t othello .
# Frontend (serve/frontend) is served by the same process at "/".

# ---- stage 1: build Edax (Linux, x86-64 or arm64 via TARGETARCH) and fetch its eval weights ----------
FROM debian:bookworm-slim AS edax
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
        git make clang libc6-dev ca-certificates curl p7zip-full \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 https://github.com/abulmo/edax-reversi.git \
    && mkdir -p edax-reversi/bin \
    && if [ "$TARGETARCH" = "arm64" ]; then A=armv8-a; else A=x86-64; fi \
    && make -C edax-reversi/src build ARCH=$A COMP=clang OS=linux \
    && curl -L -o eval.7z https://github.com/abulmo/edax-reversi/releases/download/v4.4/eval.7z \
    && 7z x eval.7z \
    && mkdir -p /edax/data \
    && cp edax-reversi/bin/lEdax-$A /edax/edax \
    && cp data/eval.dat /edax/data/eval.dat

# ---- stage 2: python runtime ---------------------------------------------------
FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple -r deploy/requirements.txt

COPY engine engine
COPY az az
COPY opponents opponents
COPY native native
COPY serve serve
COPY deploy/entrypoint.sh /entrypoint.sh
COPY --from=edax /edax third_party/edax

# Compile the C++ engine extension for this image's Python/arch (portable flags).
RUN rm -f native/*.so native/*.dylib native/*.pyd && python native/build.py \
    && python -c "import sys; sys.path[:0]=['engine','opponents']; import native, edax; \
                  print('native:', native.reload(), native.STATUS); print('edax:', edax.is_available())"

RUN apt-get purge -y g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/config')"
ENTRYPOINT ["/entrypoint.sh"]
