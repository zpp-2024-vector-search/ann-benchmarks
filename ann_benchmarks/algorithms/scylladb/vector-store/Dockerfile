FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install build-essential pkg-config ca-certificates curl libssl-dev -y --no-install-recommends &&\
    apt-get clean

# Install rustup manually to ensure a new version of rust
RUN curl https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup install 1.87

# Install the vector store service
RUN cargo install --git https://github.com/scylladb/vector-store --rev 451c651a16f0482efc5bd31195c68ebf0d949c1e

ENTRYPOINT [ "vector-store" ]

# (informative)
EXPOSE 6080
