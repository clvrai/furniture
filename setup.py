from setuptools import find_packages, setup
from pathlib import Path


long_description = (Path(__file__).parent / "README.md").read_text()


setup(
    name="furniture",
    version="0.2",
    author="Youngwoon Lee",
    author_email="lywoon89@gmail.com",
    url="https://github.com/clvrai/furniture",
    description="IKEA furniture assembly environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "requests",
        "colorlog",
        "tqdm",
        "glfw",
        "gym",
        "imageio",
        "imageio-ffmpeg",
        "ipdb",
        "mujoco-py",
        "opencv-python",
        "pybullet==1.9.5",
        "pyquaternion",
        "hjson",
        "pyyaml",
        "gdown",
        "hydra-core",
    ],
)
