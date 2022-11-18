from setuptools import find_packages, setup

setup(
    name="Google_Dummy",
    version="1.0.0",
    description="Simplified representation of a simple DS workflow.",
    maintainer="Hendrik Hülsbusch",
    maintainer_email="hhuelsbusch@dohle.com",
    url="https://github.com/HendrikHuel/Google_Dummy",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=open("README.md").read(),
    install_requires=open("requirements.txt", "r").read().splitlines(),
    include_package_data=True,
)
