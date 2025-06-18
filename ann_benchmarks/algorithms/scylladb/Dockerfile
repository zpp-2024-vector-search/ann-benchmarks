# FROM scylladb/scylladb-releng:2025.3.0-dev-0.20250616.b9e1709b238d-x86_64

# Need to use a newer version because older ones use gcc that is too old to compile usearch
FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install python3 python3-pip python3-venv python-is-python3 build-essential pkg-config ca-certificates curl libssl-dev -y --no-install-recommends
RUN python3 --version

# Install docker
RUN install -m 0755 -d /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
RUN chmod a+r /etc/apt/keyrings/docker.asc

RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update && \
    apt-get install docker-ce-cli -y --no-install-recommends && \
    apt-get clean

# Install rustup manually to ensure a new version of rust
RUN curl https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup install 1.87

# Install the vector store service
RUN cargo install --git https://github.com/scylladb/vector-store --rev 451c651a16f0482efc5bd31195c68ebf0d949c1e

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip3 install -r requirements.txt

# Pull the image with ScyllaDB
RUN docker pull scylladb/scylladb-releng:2025.3.0-dev-0.20250616.b9e1709b238d-x86_64

ENTRYPOINT ["bash ./entrypoint.sh"]
