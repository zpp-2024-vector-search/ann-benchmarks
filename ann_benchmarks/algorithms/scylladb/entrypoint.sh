#!/usr/bin/env bash
set -e
trap 'kill -TERM $SCYLLA_PID' EXIT

docker-entrypoint.sh scylladb -R &
SCYLLA_PID=$!

echo "⏳ Waiting for Scylladb to come up …"
until cqlsh -u scylladb -p scylladb 127.0.0.1 9042 -e "DESCRIBE CLUSTER" >/dev/null 2>&1
do
    sleep 3
done
echo "✅ Scylladb is up!"

python3 -u /home/app/run_algorithm.py "$@"

wait $SCYLLA_PID