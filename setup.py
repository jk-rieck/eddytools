"""Setup eddytools."""

from setuptools import setup

setup(name='eddytools',
      description='Detect, track, sample, average eddies.',
      packages=['eddytools'],
      package_dir={'eddytools': 'eddytools'},
      install_requires=['setuptools', ],
      zip_safe=False)

