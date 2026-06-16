FROM rust:1-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10002 --create-home api
COPY --from=builder /app/target/release/honcho-api-rs /usr/local/bin/honcho-api-rs

USER api
EXPOSE 8001
ENV RUST_API_BIND_ADDRESS=0.0.0.0:8001

CMD ["honcho-api-rs"]
