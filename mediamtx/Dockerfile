FROM debian:bookworm-slim

ARG MTX_VERSION=v1.12.0
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    tar \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mediamtx

RUN set -eux; \
  case "$TARGETARCH" in \
    amd64) MTX_ARCH=amd64 ;; \
    arm64) MTX_ARCH=arm64 ;; \
    *) echo "unsupported architecture: $TARGETARCH" && exit 1 ;; \
  esac; \
  curl -fsSL -o /tmp/mediamtx.tar.gz \
    "https://github.com/bluenviron/mediamtx/releases/download/${MTX_VERSION}/mediamtx_${MTX_VERSION}_linux_${MTX_ARCH}.tar.gz"; \
  tar -xzf /tmp/mediamtx.tar.gz -C /usr/local/bin mediamtx; \
  rm -f /tmp/mediamtx.tar.gz; \
  chmod +x /usr/local/bin/mediamtx

COPY mediamtx.yml /mediamtx.yml

EXPOSE 8554 8888 8889 9997 9998 9999 1935 1936 8000 8001 8189

ENTRYPOINT ["/usr/local/bin/mediamtx", "/mediamtx.yml"]
