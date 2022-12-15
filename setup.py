from setuptools import setup, find_packages

setup(
    name='age_estimation_ldl_pytorch',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)