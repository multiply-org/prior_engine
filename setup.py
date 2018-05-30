#!/usr/bin/env python

from setuptools import setup

requirements = [
    'python-dateutil',
    'gdal',
    'matplotlib',
    'numpy',
    'pyyaml',
    'shapely'
]

__version__ = None
with open('multiply_prior_engine/version.py') as f:
    exec(f.read())

setup(name='multiply-prior-engine',
      version=__version__,
      description='MULTIPLY Prior Engine',
      author='MULTIPLY Team',
      packages=['multiply_prior_engine'],
      entry_points={
          'prior_creators': [
              'vegetation_prior_creator = multiply_prior_engine:vegetation_prior_creator.VegetationPriorCreator',
              'soil_moisture_prior_creator = multiply_prior_engine:soilmoisture_prior_creator.SoilMoisturePriorCreator',
              'roughness_prior_creator = multiply_prior_engine:soilmoisture_prior_creator.RoughnessPriorCreator',
          ],
      },
      install_requires=requirements
)
