import uuid
from time import sleep
from cassandra.cluster import Cluster, NoHostAvailable
from cassandra.query import BatchStatement, BatchType
from cassandra.cluster import OperationTimedOut
from cassandra import WriteTimeout

from ann_benchmarks.algorithms.base.module import BaseANN
import subprocess

# SCYLLA_DOCKER_IMAGE_TAG = "scylladb/scylladb-releng:2025.3.0-dev-0.20250616.b9e1709b238d-x86_64"
SCYLLA_DOCKER_IMAGE_TAG = "my-vector-search-scylla"
SCYLLA_DOCKER_CONTAINER_NAME = "my-vector-search-scylla-container"

VECTOR_STORE_IMAGE_TAG = "vector-store"
VECTOR_STORE_CONTAINER_NAME = "vector-store-container"

RETRY_LIMIT   = 10         # max attempts per batch (1 original + 2 retries)v
BACKOFF_START = 0.5        # seconds; doubled after every retry

class Scylladb(BaseANN):
    def __init__(self, metric, dimension, method_param):
        print("==== METHOD ====", method_param)
        self.metric = metric
        self.dimension = dimension
        self.method_param = method_param
        self.keyspace = "ann_benchmarks"
        self.param_string = "_".join(k + "_" + str(v) for k, v in self.method_param.items()).lower()
        self.index_name = f"os_{self.param_string}"
        self.table_name = f"vector_store_{self.param_string}"
        self.name = f"Scylladb {self.param_string}"
        self.vector_store = None
        self.cluster = None
        self.conn = None

        self._setup_scylla()
        self._setup_keyspace()

    def _setup_scylla(self):
        print("Starting ScyllaDB")
        subprocess.run(["docker", "compose", "up", "scylladb", "-d"]).check_returncode()
        # TODO: Capture the ID of the container and use it to stop it later
        # subprocess.run(["docker", "run", 
        #                 "-p", "9042:9042",
        #                 "-p", "19042:19042",
        #                 "--detach", "--name", SCYLLA_DOCKER_CONTAINER_NAME, SCYLLA_DOCKER_IMAGE_TAG]).check_returncode()
        # Wait until it is brought up
        while True:
            try:
                self.cluster = Cluster(['localhost'])
                self.conn = self.cluster.connect()
                print("Successfully connected to ScyllaDB")
                return
            except NoHostAvailable as e:
                print(f"Got an error while trying to connect: {e}. Will retry")
            sleep(1)

    def _cleanup_scylla(self):
        subprocess.run(["docker", "compose", "stop", "scylladb"])

    def _setup_vector_store(self):
        print("Starting vector-store")
        subprocess.run(["docker", "compose", "up", "vector-store", "-d"]).check_returncode()
        # subprocess.run(["docker", "run",
        #                "-p", "6080:6080",
        #                "--detach", "--name", VECTOR_STORE_CONTAINER_NAME, VECTOR_STORE_IMAGE_TAG]).check_returncode()
        # try:
        #     print("Starting vector store process...")
        #     self.vector_store = subprocess.Popen(["vector-store"])
        #     print("Vector store process started.")
        # except Exception as e:
        #     raise RuntimeError("Failed to start vector store process") from e

    def _cleanup_vector_store(self):
        subprocess.run(["docker", "compose", "stop", "vector-store"])
        # subprocess.run(["docker", "stop", VECTOR_STORE_CONTAINER_NAME])
        # if self.vector_store is not None:
        #     self.vector_store.kill()
        #     self.vector_store.wait()
        #     self.vector_store = None

    def _setup_keyspace(self):
        print(f"Creating keyspace {self.keyspace}")
        self.conn.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {self.keyspace}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}};
        """)
        self.conn.set_keyspace(self.keyspace)

    def _create_table(self):
        print(f"Creating table {self.table_name}")
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id BIGINT PRIMARY KEY,
                embedding VECTOR<FLOAT, {self.dimension}>,
            ) WITH compaction = {{ 'class': 'LeveledCompactionStrategy' }};
        """)

    def _create_index(self, ):
        print(f"Creating index {self.index_name}")
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.index_name}
            ON {self.keyspace}.{self.table_name} (embedding) USING 'vector_index';
        """)
    
    def _execute_with_retry(self, statement):
        """
        Execute `statement`, transparently retrying on coordinator time-out
        up to RETRY_LIMIT times with exponential back-off.
        """
        delay   = BACKOFF_START
        attempt = 0

        print(f"Flushing a batch of {len(statement)} rows")

        while True:
            try:
                return self.conn.execute(statement)

            except (WriteTimeout, OperationTimedOut) as ex:
                print(f"Operation timed out - retrying, attempt {attempt}.")
                if attempt >= RETRY_LIMIT:
                    raise   # bubble up after exhausting retries
                attempt += 1
                sleep(delay)
                delay *= 2   # exponential back-off

    def fit(self, X, batch_size=1000):
        self.vector_dim = X.shape[1]
        self._create_table()
        self._create_index()

        print(f"Populating the table {self.table_name} with {len(X)} rows")

        insert_query = f"INSERT INTO {self.table_name} (id, embedding) VALUES (?, ?)"
        prepared = self.conn.prepare(insert_query)
        batch = BatchStatement(batch_type=BatchType.UNLOGGED)

        for i, vec in enumerate(X):
            batch.add(prepared, (i, vec.tolist()))

            if len(batch) >= batch_size:
                print(f"{i}/{len(X)}")
                self._execute_with_retry(batch)
                batch.clear()
        if batch:
            self._execute_with_retry(batch)

        # FIXME: vector store doesn't seem to be handling a situation
        # where new schema items are created too well, therefore we
        # create the vector store here and let it populate data via a scan
        self._setup_vector_store()

        # Give some time for the vector store to index data from ScyllaDB
        # FIXME: ask the vector store to explicitly wait until that happens
        print("Waiting so that the vector store service has a chance to index data")
        sleep(120)

    def set_query_arguments(self, params):
        self._ef_search = params

    def query(self, v, n):
        query = f"""
        SELECT id FROM {self.table_name}
        ORDER BY embedding ANN OF %s
        LIMIT %s
        """
        results = self.conn.execute(query, (v.tolist(), n))
        return [row.id for row in results]
   
    def batch_query(self, X, n): 
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        return self.batch_res

    def done(self):
        self._cleanup_vector_store()
        self._cleanup_scylla()
