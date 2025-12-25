"""
WarehouseBenchmark Setup Configuration

A comprehensive reinforcement learning benchmark for autonomous navigation
in warehouse environments using NVIDIA Isaac Lab and Isaac Sim.
"""

import itertools
from setuptools import setup, find_packages

# Read requirements from requirements.txt
def load_requirements(filename='requirements.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        return [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]

INSTALL_REQUIRES = load_requirements()

# Optional dependencies for advanced features
EXTRAS_REQUIRE = {
    "rsl_rl": ["rsl-rl-lib@git+https://github.com/leggedrobotics/rsl_rl.git"],
    "dev": ["pytest", "black", "flake8", "mypy"],
}

# Cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))

setup(
    name="warehouse_benchmark",
    version="0.1.0",
    author="WarehouseBenchmark Team",
    description="Reinforcement learning benchmark for warehouse navigation with Isaac Lab",
    long_description=open("README.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/WarehouseBenchmark",
    packages=find_packages(include=["source", "source.*"]),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
