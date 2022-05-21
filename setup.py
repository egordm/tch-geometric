import os

from setuptools import setup
from setuptools_rust import RustExtension

setup(
    rust_extensions=[
        RustExtension(
            "tch_geometric.tch_geometric",
            debug=os.environ.get("BUILD_DEBUG") == "1",
        )
    ],
)