FROM rust:1-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10002 --create-home api
# The release build produces two binaries from the same crate: the API server
# (honcho-api-rs) and the queue worker (honcho-deriver). Ship both so the
# compose `deriver` service can reuse this image with a different command.
COPY --from=builder /app/target/release/honcho-api-rs /usr/local/bin/honcho-api-rs
COPY --from=builder /app/target/release/honcho-deriver /usr/local/bin/honcho-deriver

USER api
EXPOSE 8001
ENV RUST_API_BIND_ADDRESS=0.0.0.0:8001

CMD ["honcho-api-rs"]
