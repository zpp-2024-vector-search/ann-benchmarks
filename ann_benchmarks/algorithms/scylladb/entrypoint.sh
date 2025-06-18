#!/usr/bin/env bash
set -e

DOCKER_IMAGE=scylladb/scylladb-releng:2025.3.0-dev-0.20250616.b9e1709b238d-x86_64

echo "⏳ Waiting for Scylladb to come up …"
until cqlsh -u scylladb -p scylladb 127.0.0.1 9042 -e "DESCRIBE CLUSTER" >/dev/null 2>&1
do
    sleep 3
done
echo "✅ Scylladb is up!"

python3 -u /home/app/run_algorithm.py "$@"
