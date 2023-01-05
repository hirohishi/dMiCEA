from setuptools import setup

setup(
    name="dMicEA",
    version="0.0.1",
    author="Hiroaki Ohishi",
    author_email = "hirohishi@outlook.jp",
    install_requires=[],
    entry_points={
        "console_scripts": [
            "dMicEA = src:cli"
        ]
    }
)
