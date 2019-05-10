import typing
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pyspark
from pyspark.sql import SparkSession, functions as f
from pyspark.sql.types import DoubleType, LongType, StructField, StructType
import xarray as xr

from dltools import SpkHits
from dltools.cov import cov3d_complicated


builder = (
    SparkSession
    .builder
    .config("spark.executor.memory", "16g")
    .config("spark.driver.memory", "16g")
    .config("spark.driver.maxResultSize", "4g")
    .config(
        "spark.jars.packages",
        "org.mongodb.spark:mongo-spark-connector_2.11:2.4.0,"
        "org.diana-hep:spark-root_2.11:0.1.15,"
    )
)


def run(
        intfr: float,
        intto: float,
        p: float = None,
        n: int = None,
        **kwargs,
        ) -> xr.Dataset:
    if len(kwargs) != 0:
        raise ValueError("Please give arguments with keywords.")
    if p is None and n is None:
        p = 1
    if p is not None and p < 1:
        raise ValueError("Argument p has to be larger than 1.")
    if p is not None and n is not None:
        raise ValueError("Cannot take both argument n and p.")

    with builder.getOrCreate() as spark:
        df = (
            spark
            .read
            .format("com.mongodb.spark.sql.DefaultSource")
            .option("uri", "mongodb://mongodb/sacla_2017b8065.resorted")
            .option("pipeline", """[
                {{
                    $match: {{aq: {{$in: [28, 29, 35, 36]}},
                              gm2: {{$gte: {fr}, $lt: {to}}}}},
                }},
                {{
                    $project: {{
                        tag: true,
                        aq: true,
                        gm2: true,
                        hits: {{
                            $filter: {{
                                input: "$hits",
                                as: "item",
                                cond: {{
                                    $and: [
                                        {{$lte: ["$$item.flag", 6]}},
                                        {{$or: [{{$ne: ["$$item.as_.C1p", undefined]}},
                                                {{$ne: ["$$item.as_.I2p", undefined]}},
                                                {{$ne: ["$$item.as_.I1p", undefined]}}]}},
                                    ],
                                }},
                            }},
                        }},
                    }},
                }},
            ]""".format(fr=intfr, to=intto))
            .schema(
                StructType([
                    StructField("tag", LongType()),
                    StructField("aq", LongType()),
                    StructField("gm2", DoubleType()),
                    StructField("hits", SpkHits)
                ])
            )
            .load()
        )

        def ishit0(hit: pyspark.sql.Row) -> bool:
            return "C1p" in hit["as_"]

        def ishit1(hit: pyspark.sql.Row) -> bool:
            return "I2p" in hit["as_"]

        def ishit2(hit: pyspark.sql.Row) -> bool:
            return "I1p" in hit["as_"]

        m = df.count()
        if n is None:
            n = p * m
        elif n < m:
            n = m

        combined = cov3d_complicated(
            df, ishit0, ishit1, ishit2,
            npart=12,
            opt1={
                "npart": 12,
                "fraction": 1,
            },
            opt2={
                # "npart": 60,
                "fraction": min(n ** (1/2) / m, 1),
            },
            opt3={
                # "npart": 300,
                "fraction": min(n ** (1/3) / m, 1),
            },
        )

        def hist(
                rows: typing.List[pyspark.sql.Row]
                ) -> typing.List[xr.Dataset]:
            filtered = [
                {
                    "pz": (r["hit0"]["as_"]["C1p"]["pz"]
                           + r["hit1"]["as_"]["I2p"]["pz"]
                           + r["hit2"]["as_"]["I1p"]["pz"]),
                    "px": (r["hit0"]["as_"]["C1p"]["px"]
                           + r["hit1"]["as_"]["I2p"]["px"]
                           + r["hit2"]["as_"]["I1p"]["px"]),
                    "py": (r["hit0"]["as_"]["C1p"]["py"]
                           + r["hit1"]["as_"]["I2p"]["py"]
                           + r["hit2"]["as_"]["I1p"]["py"]),
                    "ke": (r["hit0"]["as_"]["C1p"]["ke"]
                           + r["hit1"]["as_"]["I2p"]["ke"]
                           + r["hit2"]["as_"]["I1p"]["ke"]),
                }
                for r in rows
                if (-200 < (r["hit0"]["as_"]["C1p"]["px"]
                            + r["hit1"]["as_"]["I2p"]["px"]
                            + r["hit2"]["as_"]["I1p"]["px"]) < 200
                    and -200 < (r["hit0"]["as_"]["C1p"]["py"]
                                + r["hit1"]["as_"]["I2p"]["py"]
                                + r["hit2"]["as_"]["I1p"]["py"]) < 200
                    and -200 < (r["hit0"]["as_"]["C1p"]["pz"]
                                + r["hit1"]["as_"]["I2p"]["pz"]
                                + r["hit2"]["as_"]["I1p"]["pz"]) < 200)
            ]
            edges_p = np.linspace(-200, 200, 81)  # bin size: 5
            edges_k = np.linspace(0, 100, 101)  # bin size: 1
            pz, *_ = np.histogram(
                [r["pz"] for r in filtered],
                bins=edges_p,
            )
            px, *_ = np.histogram(
                [r["px"] for r in filtered],
                bins=edges_p,
            )
            py, *_ = np.histogram(
                [r["py"] for r in filtered],
                bins=edges_p,
            )
            ke, *_ = np.histogram(
                [r["ke"] for r in filtered],
                bins=edges_k,
            )
            return [xr.Dataset(
                {"pz": ("edge_p", pz),
                 "px": ("edge_p", px),
                 "py": ("edge_p", py),
                 "ker": ("edge_ker", ke)},
                coords={"edge_p": edges_p[:-1], "edge_ker": edges_k[:-1]}
            )]
        return combined(hist)


step = 0.002
for fr in np.arange(0.005, 0.021, step):
    to = fr + step
    filename = ("Data/Cov KER; "
                "target=C1p,I2p,I1p "
                "& flag<=6 "
                "& p=-200--200 "
                f"& gm2={fr:.3f}--{to:.3f}.pickle")
    if os.path.exists(filename):
        continue

    d = run(intfr=fr, intto=to, n=1000000000)
    with open(filename, "wb") as fp:
        pickle.dump(d, fp)
