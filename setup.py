from setuptools import find_packages, setup


setup(
    name="furniture",
    version="0.2",
    author="Youngwoon Lee",
    author_email="lywoon89@gmail.com",
    description="IKEA furniture assembly environment",
    url="https://github.com/clvrai/furniture",
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
        "openvr",
        "pybullet==1.9.5",
        "pyquaternion",
        "hjson",
        "pyyaml",
        "gdown",
    ],
)
