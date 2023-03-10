from setuptools import setup, find_packages

setup(
    name='age-estimation-ldl-pytorch',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'tqdm',
        'pretrainedmodels',
        'albumentations',
        'facenet_pytorch',
        'pandas',
        'Pillow',
        'matplotlib'

    ],
)