# TODO: Use a dedicated docker container for the vector store. For now, the dockerfile is a little bit
# hacky as it needs to use a particular version of ubuntu, needs to install everything etc.
#
# Need to use a newer version because older ones use gcc that is too old to compile usearch
FROM ann-benchmarks

# Add Docker's official GPG key:
RUN apt-get update
RUN apt-get install ca-certificates curl
RUN install -m 0755 -d /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
RUN chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update 
RUN apt-get install -y wget docker-ce-cli docker-compose-plugin

RUN pip3 install scylla-driver

COPY ./ann_benchmarks/algorithms/scylladb/docker-compose.yml .

ENTRYPOINT ["python", "-u", "run_algorithm.py"]
