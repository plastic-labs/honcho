FROM rust:1-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10003 --create-home worker
COPY --from=builder /app/target/release/honcho-worker-rs /usr/local/bin/honcho-worker-rs

USER worker

CMD ["honcho-worker-rs"]
