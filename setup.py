from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


setup(
    name='macsim',
    version='1.0',
    python_requires='~=3.10', # we need at least python 3.10 for new dataclass features
    py_modules=[],
    install_requires=parse_requirements('requirements.txt'),
    packages=find_packages('.')
)