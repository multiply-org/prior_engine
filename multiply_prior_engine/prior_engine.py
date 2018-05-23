#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Prior Engine for MULTIPLY.

    Copyright (C) 2018  Thomas Ramsauer
"""

import logging
import os
import pdb
import sys

import yaml

from .soilmoisture_prior import RoughnessPrior, SoilMoisturePrior
from .vegetation_prior import VegetationPrior


__author__ = ["Alexander Löw", "Thomas Ramsauer"]
__copyright__ = "Copyright 2018, Thomas Ramsauer"
__credits__ = "Alexander Löw"
__maintainer__ = "Thomas Ramsauer"
__email__ = "t.ramsauer@iggf.geo.uni-muenchen.de"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler(__name__ + '.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# to show output in console. set to higher level to omit.
# Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
stream_handler.setLevel(logging.ERROR)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def _get_config(configfile):
    """
    Load config from self.configfile.
    writes to self.config.

    :returns: -
    """
    try:
        with open(configfile, 'r') as cfg:
            config = yaml.load(cfg)
    except FileNotFoundError as e:
        logger.INFO('Info: current directory: {}'.format(os.getcwd()))
        logger.ERROR('{}'.format(e.args[0]))
        raise
    try:
        assert 'Prior' in config.keys(),\
            ('There is no prior section in configuration file ({}).'
             .format(configfile))
        assert config['Prior'] is not None,\
            ('There is no prior configuration in the config file ({}).'
             .format(configfile))
    except AssertionError as e:
        logger.ERROR('{}'.format(e.args[0]))
        raise
    return config


default_config = os.path.join(os.path.dirname(__file__),
                              'sample_config_prior.yml')


class PriorEngine(object):
    """ Prior Engine for MULTIPLY.

        holds prior initialization methods (e.g. config loading).
        calls specific submodules (soilmoisture_prior, vegetation_prior, ..)
    """

    # TODO ad correct sub routines from Joris
    subengine = {
        'sm': SoilMoisturePrior,
        'dielectric_const': '',
        'roughness': RoughnessPrior,
        'lai': VegetationPrior,
        'cab': VegetationPrior,
        'car': VegetationPrior,
        'cdm': VegetationPrior,
        'cw': VegetationPrior,
        'N': VegetationPrior
    }

    def __init__(self, **kwargs):
        self.configfile = None
        while self.configfile is None:
            self.configfile = kwargs.get('config', None)
            self.configfile = kwargs.get('configfile', None)
            # have a backup/default config:
            self.configfile = default_config
        assert os.path.exists(self.configfile)
        self.datestr = kwargs.get('datestr', None)
        self.variables = kwargs.get('variables', None)
        # self.priors = self.config['Prior']['priors']
        # TODO get previous state.
        # TODO get subengines

        self.config = _get_config(self.configfile)
        self._check()
        logger.INFO('Loaded {}.'.format(self.configfile))

    def _check(self):
        """initial check for passed values of
        - config
        - datestr
        - variables

        :returns: -
        :rtype: -

        """
        try:
            assert self.config is not None, \
               'Could not load configfile.'
            # assert self.priors is not None, \
            #     'There is no prior specified in configfile.'
            assert self.datestr is not None, \
                'There is no date passed to the Prior Engine.'
            assert self.variables is not None, \
                'There are no variables for prior retrieval specified on.'
            # TODO Should previous state be integrated here?
            logger.DEBUG('Loaded config:\n{}.'.format(self.config))
        except AssertionError as e:
            logger.ERROR('{}'.format(e.args[0]))
            raise

    def get_priors(self):
        """
        Get prior data.
        calls *_get_prior* for all variables (e.g. sm, lai, ..) passed on to
        get_mean_state_vector method.

        :returns: dictionary with prior names/prior types/filenames as
                  {key/{key/values}}.
        :rtype: dictionary of dictionary

        """
        res = {}
        for var in self.variables:
            res.update({var: self._get_prior(var)})
        # return self._concat_priors(res)
        return res

    def _get_prior(self, var):
        """ Called by get_priors for all variables to be inferred.\
        For specific variable/prior (e.g. sm climatology) get prior\
        info and calculate/provide prior.

        :param var: prior name (e.g. sm, lai, ..)
        :returns: -
        :rtype: -

        """
        var_res = {}
        try:
            assert var in self.config['Prior'].keys(), \
                'Variable to be inferred not in config.'
            assert var in self.subengine,\
                ('No sub-enginge defined for variable to be inferred ({}).'
                .format(var))
        except AssertionError as e:
            logger.ERROR('{}'.format(e.args[0]))
            raise
        logger.info('Getting prior for variable *{}*.'.format(var))

        # test if prior type is specified (else return empty dict):
        try:
            self.config['Prior'][var].keys() is not None
        except AttributeError as e:
            logger.warning('[WARNING] No prior type for {}'
                           ' prior specified!'.format(var))
            return

        # Run subengine for all prior types of variable:
        for ptype in self.config['Prior'][var].keys():
            # pass config and prior type to subclass/engine
            try:
                logger.info('Initializing {} for {} {} prior:'
                            .format(self.subengine[var], var, ptype))
                # initialize specific prior *Class Object*
                # e.g. VegetationPrior as 'prior':
                prior = self.subengine[var](ptype=ptype, config=self.config,
                                            datestr=self.datestr, var=var)
                # call RetrievePrior from specific prior class:
                var_res.update({ptype: prior.RetrievePrior()})

            # Assertions in subengine are passed on here:
            # e.g. If no file is found: module should throw AssertionError
            except AssertionError as e:
                logger.error('{}: {}'.format(self.subengine[var], e.args[0]))
                raise
            # for now catch all built-in exceptions
            except Exception as e:
                logger.error('{}: {}'.format(self.subengine[var], e.args[0]))
                raise

        return var_res

    def _concat_priors(self, prior_dict):
        """ Concatenate individual state vectors and covariance matrices
        for sm, veg, ..

        :returns: dictionary with keys beeing superordinate prior name (sm, ..)
        :rtype: dictionary

        """
        # input: dictionary from getpriors
        # all_priors = np.concatenate((p, std), axis=0)
        # all_cov = np.concatenate((p, std), axis=0)
        res_concat = {}
        for key in self.config['Prior'].keys():
            if key == 'priors':
                continue
            # concatenate all values v for keys k if k contains key from
            # self.config Prior keys('sm')
            temp_dict = {k: v for (k, v) in prior_dict.items() if key in k}
            res_concat.update({key: list(zip(temp_dict.values()))})

        return res_concat


def get_mean_state_vector(datestr: str, variables: list,
                          config: str="./sample_config_prior.yml") -> dict:
    """
    Return dictionary with variable dependent sub dictionary with prior type
    (key) and filenames of prior files (values).

    :param datestr: The date (time?) for which the prior needs to be derived
    :param variables: A list of variables (sm, lai, roughness, ..)
    for which priors need to be available

    :return: dictionary with keys being the variables and
    values being a dictionary of prior type and filename of prior file.
    """

    return (PriorEngine(datestr=datestr, variables=variables,
                        config=config)
            .get_priors())


if __name__ == '__main__':
    print(get_mean_state_vector(
        datestr="2017-03-01", variables=['sm', 'lai', 'cab']))
