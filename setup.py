from setuptools import setup, find_packages

setup(
    name='aqua',
    version='0.0.1',
    packages=find_packages(),
    description='Python module to allocate responsive offloaded tensors for generative inference.',
    author='Abhishek Vijaya Kumar',
    author_email='abhivijay96@gmail.com',
    install_requires=[
        'torch',
    ]
)
