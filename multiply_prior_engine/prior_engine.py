#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Prior Engine for MULTIPLY.

    Copyright (C) 2018  Thomas Ramsauer
"""

import logging
import pkg_resources
import os
import pdb
import sys

import yaml

from .soilmoisture_prior_creator import RoughnessPriorCreator, SoilMoisturePriorCreator
from .vegetation_prior_creator import VegetationPriorCreator


__author__ = ["Alexander Löw", "Thomas Ramsauer"]
__copyright__ = "Copyright 2018, Thomas Ramsauer"
__credits__ = "Alexander Löw"
__maintainer__ = "Thomas Ramsauer"
__email__ = "t.ramsauer@iggf.geo.uni-muenchen.de"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S %p')

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
        logger.info('Info: current directory: {}'.format(os.getcwd()))
        logger.error('{}'.format(e.args[0]))
        raise
    try:
        assert 'Prior' in config.keys(),\
            ('There is no prior section in configuration file ({}).'
             .format(configfile))
        assert config['Prior'] is not None,\
            ('There is no prior configuration in the config file ({}).'
             .format(configfile))
    except AssertionError as e:
        logger.error('{}'.format(e.args[0]))
        raise
    return config


default_variables = ['Cab',
                     'Car',
                     'Cdm',
                     'Cb',
                     'Cw',
                     'N',
                     'Albedo',
                     'LAI',
                     'Fapar',
                     'VWC',
                     'LIDFa',
                     'H',
                     'Bsoil',
                     'Psoil',
                     'SM',
                     'SR']
default_variables_lower = [x.lower() for x in default_variables]


class PriorEngine(object):
    """ Prior Engine for MULTIPLY.

        holds prior initialization methods (e.g. config loading).
        calls specific submodules (soilmoisture_prior, vegetation_prior, ..)
    """

    default_config = os.path.join(os.path.dirname(__file__),
                                  'sample_config_prior.yml')

    def __init__(self, **kwargs):
        self.configfile = None
        self.configfile = kwargs.get('config', None)
        if self.configfile is None:
            self.configfile = kwargs.get('configfile', None)
        if self.configfile is None:
            # have a backup/default config:
            logger.warning('Using default config file {}. No keyword argument '
                           'found while initializing SoilMoisturePriorCreator.'
                           .format(self.default_config))
            self.configfile = self.default_config
        print('Using config file: {}'.format(self.configfile))
        assert os.path.exists(self.configfile)
        self.datestr = kwargs.get('datestr', None)
        self.variables = kwargs.get('variables', None)
        # self.priors = self.config['Prior']['priors']

        # TODO ad correct sub routines from Joris

        self.subengine = {}
        logger.info('Loading sub-engines for variables.')
        prior_creator_registrations = list(pkg_resources.iter_entry_points(
                                           'prior_creators'))
        logger.info('Got following prior_creator_registrations from entry '
                    'points: {}'.format(prior_creator_registrations))
        for prior_creator_registration in prior_creator_registrations:
            prior_creator = prior_creator_registration.load()
            variable_names = prior_creator.get_variable_names()
            for variable_name in variable_names:
                self.subengine[variable_name] = prior_creator
                logger.info('Sub-engine for {}: {}.'
                            .format(variable_name, prior_creator))
        logger.info('Got sub-engines for {}.'
                    .format([k for k in self.subengine.keys()]))

        self.config = _get_config(self.configfile)
        self._check()
        logger.info('Loaded {}.'.format(self.configfile))

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
            assert len(self.subengine) > 0, \
                'There is no sub-engine specified in the Prior Engine.'
            assert self.variables is not None, \
                'There are no variables for prior retrieval specified on.'
            # TODO Should previous state be integrated here?
            logger.debug('Loaded config:\n{}.'.format(self.config))
        except AssertionError as e:
            logger.error('{}'.format(e.args[0]))
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
                ('No sub-engine defined for variable to be inferred ({}).\n'
                 'Current sub-engines:\n{}'
                 .format(var, self.subengine))
        except AssertionError as e:
            logger.error('{}'.format(e.args[0]))
            raise
        logger.info('Getting prior for variable *{}*.'.format(var))

        # test if prior type is specified (else return empty dict):
        try:
            self.config['Prior'][var].keys() is not None
        except AttributeError:
            logger.warning('[WARNING] No prior type for {} prior specified!'
                           .format(var))
            return
        # fill variable specific dictionary with all priors (clim, recent, ..)
        # TODO concatenation of prior files
        # be returned instead/as additional form
        for ptype in self.config['Prior'][var].keys():

            # pass config and prior type to subclass/engine
            try:
                logger.info('Initializing {} for {} {} prior:'
                            .format(self.subengine[var], var, ptype))
                # initialize specific prior *Class Object*
                # e.g. VegetationPrior as 'prior':
                prior = self.subengine[var](ptype=ptype, config=self.config,
                                            datestr=self.datestr, var=var)
                var_res.update({ptype: prior.compute_prior_file()})

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
