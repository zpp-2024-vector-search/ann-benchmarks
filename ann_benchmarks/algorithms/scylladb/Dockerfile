FROM scylladb/scylla:6.2.2

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/app
COPY requirements.txt .

COPY run_algorithm.py .
COPY ann_benchmarks/algorithms/scylladb/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 9042

ENTRYPOINT ["entrypoint.sh"]
