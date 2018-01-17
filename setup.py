#!/usr/bin/env python

from setuptools import setup

requirements = [
    'nose',
    'shapely',
    'pyyaml',
    'pytest', 'nose',
    'shapely',
    'pyyaml',
    'datetime',
    'os',
    're',
    'numpy',
    'shapely',
    'netCDF4',
    'scipy',
    'yaml',
    'dateutil'
]

setup(name='multiply-prior-engine',
      version='0.1',
      description='MULTIPLY Prior Engine',
      author='MULTIPLY Team',
      packages=['multiply_prior_engine'],
      entry_points={
          'file_system_plugins': [
              'local_file_system = multiply_data_access:local_file_system.LocalFileSystemAccessor',
          ],
          'meta_info_provider_plugins': [
              'json_meta_info_provider = multiply_data_access:json_meta_info_provider.JsonMetaInfoProviderAccessor',
          ],
      },
      install_requires=requirements
)
