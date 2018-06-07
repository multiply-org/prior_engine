#!/usr/bin/env python

from setuptools import setup

with open('requirements.txt') as r:
    requirements = r.read().splitlines()

__version__ = None
__status__ = None
__license__ = None

with open('multiply_prior_engine/version.py') as f:
    exec(f.read())

setup(name='multiply-prior-engine',
      version=__version__,
      description='MULTIPLY Prior Engine',
      author='MULTIPLY Team',
      packages=['multiply_prior_engine'],
      entry_points={
          'console_scripts': [
              'user_prior = multiply_prior_engine.user_prior:main'
          ]
      },
      install_requires=requirements
      )
