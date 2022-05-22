import os

from setuptools import setup
from setuptools_rust import RustExtension
import toml

version = toml.load('Cargo.toml')['package']['version']
if os.getenv('VERSION_SUFFIX', False):
    version = f'{version}-{os.getenv("VERSION_SUFFIX")}'

setup(
    rust_extensions=[
        RustExtension(
            "tch_geometric.tch_geometric",
            debug=os.environ.get("BUILD_DEBUG") == "1",
        )
    ],
    version=version
)
