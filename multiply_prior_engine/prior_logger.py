#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Logger for the Prior Engine for MULTIPLY.

    Copyright (C) 2018  Thomas Ramsauer
"""

import os
import yaml
import logging
import logging.config


__author__ = "Thomas Ramsauer"
__copyright__ = "Thomas Ramsauer"
__maintainer__ = "Thomas Ramsauer"
__email__ = "t.ramsauer@iggf.geo.uni-muenchen.de"


class PriorLogger(object):
    """ Logger for the Prior Engine for MULTIPLY.

    initializes logging
    """

    configfile = os.path.join(os.path.dirname(__file__),
                              'prior_engine_logging.yml')

    def __init__(self):
        with open(self.configfile, 'r') as cfg:
            config_dict = yaml.load(cfg)
        logging.config.dictConfig(config_dict)
        logger = logging.getLogger(__name__)
        logger.info('Logger initialized.')

    def logger(self):
        return logging.getLogger(__name__)
