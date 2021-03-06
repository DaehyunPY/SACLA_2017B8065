import os
import pickle
import typing

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import LongType, DoubleType, StructType, StructField
import numpy as np
import xarray as xr

from dltools import SpkHits
from dltools.cov import cov3d_complicated


builder = (
    SparkSession
    .builder
    .config("spark.executor.memory", "12g")
    .config("spark.driver.memory", "12g")
    .config("spark.driver.maxResultSize", "8g")
    .config(
        "spark.jars.packages",
        "org.mongodb.spark:mongo-spark-connector_2.11:2.4.0,"
        "org.diana-hep:spark-root_2.11:0.1.15,"
    )
)


k0, k1, k2 = "C1p", "I2p", "I1p"


def norm(d: dict) -> float:
    return sum(d[v] ** 2 for v in ("pz", "px", "py")) ** 0.5


def unit(d: dict) -> dict:
    pr = norm(d)
    return {v: d[v] / pr for v in ("pz", "px", "py")}


def inner(a: dict, b: dict) -> float:
    return sum(a[v] * b[v] for v in ("pz", "px", "py"))


def cross(a: dict, b: dict) -> dict:
    product = np.cross([a["pz"], a["px"], a["py"]],
                       [b["pz"], b["px"], b["py"]])
    return dict(zip(("pz", "px", "py"), product))


def angle_btw(a: dict, b: dict) -> float:
    return inner(a, b) / norm(a) / norm(b)


def transfer(d: dict, frame: dict) -> dict:
    return {
        **d,
        **{v: inner(d, frame[v]) for v in ("pz", "px", "py")},
    }


def run(intfr: float, intto: float,
        p: float = None, n: int = None,
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
                        hits: {{
                            $filter: {{
                                input: "$hits",
                                as: "item",
                                cond: {{
                                    $and: [
                                        {{$lte: ["$$item.flag", 14]}},
                                        {{$or: [{keys}]}},
                                    ],
                                }},
                            }},
                        }},
                    }},
                }},
            ]""".format(fr=intfr, to=intto,
                        keys=",".join('{{$ne: ["$$item.as_.{}", undefined]}}'.format(k)
                                      for k in {k0, k1, k2})))
            .schema(
                StructType([
                    StructField("hits", SpkHits),
                ])
            )
            .load()
        )

        def ishit0(hit: Row) -> bool:
            return k0 in hit["as_"]

        def ishit1(hit: Row) -> bool:
            return k1 in hit["as_"]

        def ishit2(hit: Row) -> bool:
            return k2 in hit["as_"]

        m = df.count()
        if n is None:
            n = p * m
        elif n < m:
            n = m

        combined = cov3d_complicated(
            df, ishit0, ishit1, ishit2,
            npart=24,
            opt1={"fraction": 1},
            opt2={"fraction": min(n ** (1/2) / m, 1)},
            opt3={"fraction": min(n ** (1/3) / m, 1)},
        )

        def hist(
                rows: typing.List[Row]
                ) -> typing.List[xr.Dataset]:
            def gen() -> typing.Iterable[dict]:
                for row in rows:
                    r = row.asDict(recursive=True)
                    flatten = {
                        f"hit{i}": {**r[f"hit{i}"].pop("as_")[k], **r[f"hit{i}"]}
                        for i, k in enumerate((k0, k1, k2))
                    }
                    flatten["summed"] = {
                        v: sum(flatten[f"hit{i}"][v] for i in range(3))
                        for v in ("pz", "px", "py", "ke")
                    }
                    # if 200 < norm(flatten["summed"]):
                    #     continue
                    xunit = unit(flatten["hit0"])
                    zunit = unit(cross(flatten["hit0"], flatten["hit1"]))
                    yunit = cross(zunit, xunit)
                    frame = {"pz": zunit, "px": xunit, "py": yunit}
                    yield {
                        "inlab": flatten,
                        "inmol": {k: transfer(d, frame) for k, d in flatten.items()},
                    }

            filtered = list(gen())
            if len(filtered) == 0:
                return []

            edges = {
                "p": np.linspace(-500, 500, 201),  # bin size: 5
                "ke": np.linspace(0, 150, 151),  # bin size: 1
                "cosang": np.linspace(-1, 1, 101),  # bin size: 0.02
                "normp": np.linspace(-4, 4, 81),  # bin size: 0.1
            }
            sumpz, _ = np.histogram(
                [r["inlab"]["summed"]["pz"] for r in filtered],
                bins=edges["p"],
            )
            sumpx, _ = np.histogram(
                [r["inlab"]["summed"]["px"] for r in filtered],
                bins=edges["p"],
            )
            sumpy, _ = np.histogram(
                [r["inlab"]["summed"]["py"] for r in filtered],
                bins=edges["p"],
            )
            sumke, _ = np.histogram(
                [r["inlab"]["summed"]["ke"] for r in filtered],
                bins=edges["ke"],
            )
            h0h1angle, _ = np.histogram(
                [angle_btw(r["inlab"]["hit0"],
                           r["inlab"]["hit1"]) for r in filtered],
                bins=edges["cosang"],
            )
            h0h2angle, _ = np.histogram(
                [angle_btw(r["inlab"]["hit0"],
                           r["inlab"]["hit2"]) for r in filtered],
                bins=edges["cosang"],
            )
            h1h2angle, _ = np.histogram(
                [angle_btw(r["inlab"]["hit1"],
                           r["inlab"]["hit2"]) for r in filtered],
                bins=edges["cosang"],
            )
            pr = np.array([norm(r["inmol"]["hit0"]) for r in filtered])
            h1normpxy, _, _ = np.histogram2d(
                np.array([r["inmol"]["hit1"]["px"] for r in filtered]) / pr,
                np.array([r["inmol"]["hit1"]["py"] for r in filtered]) / pr,
                bins=2*[edges["normp"]],
            )
            h2normpxy, _, _ = np.histogram2d(
                np.array([r["inmol"]["hit2"]["px"] for r in filtered]) / pr,
                np.array([r["inmol"]["hit2"]["py"] for r in filtered]) / pr,
                bins=2*[edges["normp"]],
            )
            h2normpxz, _, _ = np.histogram2d(
                np.array([r["inmol"]["hit2"]["px"] for r in filtered]) / pr,
                np.array([r["inmol"]["hit2"]["pz"] for r in filtered]) / pr,
                bins=2*[edges["normp"]],
            )
            return [xr.Dataset(
                {"sumpz": ("p", sumpz),
                 "sumpx": ("p", sumpx),
                 "sumpy": ("p", sumpy),
                 "sumke": ("ke", sumke),
                 "h0h1angle": ("cosang", h0h1angle),
                 "h0h2angle": ("cosang", h0h2angle),
                 "h1h2angle": ("cosang", h1h2angle),
                 "h1normpxy": (("normp0", "normp1"), h1normpxy),
                 "h2normpxy": (("normp0", "normp1"), h2normpxy),
                 "h2normpxz": (("normp0", "normp1"), h2normpxz)},
                coords={"p": edges["p"][:-1],
                        "ke": edges["ke"][:-1],
                        "cosang": edges["cosang"][:-1],
                        "normp0": edges["normp"][:-1],
                        "normp1": edges["normp"][:-1]},
            )]
        return combined(hist)


step = 0.002
for fr in np.arange(0.005, 0.021, step):
    to = fr + step
    filename = ("Data/Cov ang dist and KER; "
                f"target={k0},{k1},{k2} "
                "& p=any "
                f"& gm2={fr:.3f}--{to:.3f}.pickle")
    if os.path.exists(filename):
        continue

    d = run(intfr=fr, intto=to, n=1000000000)
    with open(filename, "wb") as fp:
        pickle.dump(d, fp)
