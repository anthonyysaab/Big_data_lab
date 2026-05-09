"""
src/spark_session.py
====================
Spark session factory for the local Windows project environment.

Windows stability notes
-----------------------
Previous attempts to pre-reserve ports with Python sockets before passing
them to Spark failed because SO_REUSEADDR on Windows allows a *second* bind
to succeed (stealing the port), rather than blocking it as on Linux.  The
reservation sockets were actively preventing the JVM from binding, forcing
Spark to fall back to a different driver port while the BlockManager still
expected the original one — causing the NullPointerException in
BlockManagerMasterEndpoint.

Solution: use port 0 (OS-assigned) for both the driver and the BlockManager,
combined with maxRetries=128 and full 127.0.0.1 isolation.  On a single-node
local session the OS will assign stable ports immediately with no conflicts.

The BlockManager NPE (Cannot invoke "BlockManagerId.executorId()" because
"idWithoutTopologyInfo" is null) is a Spark 3.5 / Windows race condition
that is resolved by:
  1. Not fighting Spark over port numbers.
  2. Setting spark.local.ip in the builder (not just as an env var).
  3. Disabling the Spark UI (one fewer RPC endpoint to race against).
"""

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HADOOP_HOME = Path(r"C:\hadoop")
SPARK_NLP_PACKAGE = "com.johnsnowlabs.nlp:spark-nlp_2.12:5.3.3"


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _prepend_to_path(path: Path) -> None:
    path_str = str(path)
    current_parts = os.environ.get("PATH", "").split(os.pathsep)
    if path_str not in current_parts:
        os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")


def _configure_environment(python_executable: str) -> None:
    """
    Set every environment variable PySpark / Spark NLP needs on Windows
    before the JVM starts.
    """
    os.environ["HADOOP_HOME"] = str(HADOOP_HOME)
    os.environ["hadoop.home.dir"] = str(HADOOP_HOME)
    _prepend_to_path(HADOOP_HOME / "bin")

    # Force all Spark networking to loopback — must be set before JVM launch.
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

    os.environ["PYSPARK_PYTHON"] = python_executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _java_open_options() -> str:
    """
    --add-opens flags required by Spark / Spark NLP on Java 11+.
    """
    return " ".join([
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED",
    ])


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_spark_session(app_name: str = "big_data_nlp_project") -> SparkSession:
    """
    Create a local Spark session for the project.

    Key design decisions for Windows stability:
    - Port 0 for driver and BlockManager: let the OS assign free ports
      atomically inside the JVM.  Pre-reserving from Python causes conflicts
      on Windows due to SO_REUSEADDR semantics.
    - spark.local.ip set in the builder (not just env var): the JVM-side
      Spark config must also see 127.0.0.1 or the driver may bind on the
      wrong interface.
    - spark.ui.enabled false: removes one RPC endpoint that can race during
      BlockManager initialisation.
    - local[1]: single executor thread eliminates inter-executor RPC races.
    """
    python_executable = sys.executable
    _configure_environment(python_executable)
    java_opts = _java_open_options()

    print(f"[spark_session] Python: {python_executable}")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[1]")

        # ---- Networking (Windows loopback, no port pre-reservation) --------
        .config("spark.local.ip",            "127.0.0.1")
        .config("spark.driver.host",         "127.0.0.1")
        .config("spark.driver.bindAddress",  "127.0.0.1")
        # Port 0 = OS picks a free port atomically inside the JVM.
        # This is safe because local[1] has no separate executor process.
        .config("spark.driver.port",         "0")
        .config("spark.blockManager.port",   "0")
        .config("spark.port.maxRetries",     "128")

        # ---- Resources -----------------------------------------------------
        .config("spark.driver.memory",          "6g")
        .config("spark.driver.maxResultSize",   "1g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism",    "4")

        # ---- Execution -----------------------------------------------------
        .config("spark.sql.adaptive.enabled",           "true")
        .config("spark.serializer",                     "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max",      "1024m")
        .config("spark.sql.parquet.compression.codec",  "snappy")
        .config("spark.ui.enabled",                     "false")

        # ---- Timeouts (generous for slow local Windows JVM starts) ---------
        .config("spark.network.timeout",            "600s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.rpc.askTimeout",             "600s")

        # ---- Python env ----------------------------------------------------
        .config("spark.pyspark.python",                    python_executable)
        .config("spark.pyspark.driver.python",             python_executable)
        .config("spark.executorEnv.TF_CPP_MIN_LOG_LEVEL", "3")

        # ---- Spark NLP -----------------------------------------------------
        .config("spark.jars.packages", SPARK_NLP_PACKAGE)

        # ---- Java module access (Java 11+) ---------------------------------
        .config("spark.driver.extraJavaOptions",   java_opts)
        .config("spark.executor.extraJavaOptions", java_opts)

        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    print("[spark_session] Spark session created.")
    return spark