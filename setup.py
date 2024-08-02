from setuptools import setup, find_packages

setup(
    name='aimnet2',
    version='0.0.1',
    author='Roman Zubatyuk',
    author_email='zubatyuk@gmail.com',
    description='AIMNet2: Fast, accurate and transferable neural network interatomic potential',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'ignite',
        'omegaconf',
        'click',
        'h5py',
        'wandb'
    ],
    entry_points={
        'console_scripts': [
            'aimnet=aimnet.cli:cli'
        ],
    },    
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
)
