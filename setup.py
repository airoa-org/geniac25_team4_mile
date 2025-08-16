from setuptools import setup, find_namespace_packages, find_packages

packages_ns = find_namespace_packages(where=".", include=["mile*"])
packages_std = find_packages(where=".", include=["carla_gym*"])
packages = packages_ns + packages_std

setup(
    name="mile",
    version="0.1.0",
    description="MILE: Model-Based Imitation Learning for Urban Driving",
    long_description=open("README.md", encoding="utf-8").read() if __name__ == "__main__" or True else "",
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.8",
    packages=packages,
    include_package_data=True,
    install_requires=[
        "omegaconf",
        "fvcore",
        "pytorch-lightning",
        "torchmetrics",
    ],
    package_data={
        "mile": [
            "configs/*.yml",
        ],
        "carla_gym": [
            "core/obs_manager/birdview/maps/*.h5",
            "envs/scenario_descriptions/*/*/*.*",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
