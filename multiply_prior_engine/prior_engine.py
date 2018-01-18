#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Prior Engine for MULTIPLY.

    Copyright (C) 2018  Thomas Ramsauer
"""

import yaml

from soilmoisture_prior import RoughnessPrior, SoilMoisturePrior
from vegetation_prior import VegetationPrior

__author__ = ["Alexander Löw", "Thomas Ramsauer"]
__copyright__ = "Copyright 2018, Thomas Ramsauer"
__credits__ = ["Alexander Löw", "Thomas Ramsauer"]
__license__ = "GPLv3"
__version__ = "0.0.1"
__maintainer__ = "Thomas Ramsauer"
__email__ = "t.ramsauer@iggf.geo.uni-muenchen.de"
__status__ = "Prototype"


class PriorEngine(object):
    """
    Prior Engine for MULTIPLY.

    holds prior initialization methods (e.g. config loading).
    calls specific submodules (soilmoisture_prior, vegetation_prior, ..)
    """

    def __init__(self, **kwargs):
        self.configfile = kwargs.get('config', None)
        self.datestr = kwargs.get('datestr', None)
        self.variables = kwargs.get('variables', None)
        # self.priors = self.config['Prior']['priors']

        self._get_config()
        self._check()

    def _check(self):
        assert self.config is not None, \
            'Could not load configfile.'
        # assert self.priors is not None, \
        #     'There is no prior specified in configfile.'
        assert self.datestr is not None, \
            'There is no date passed to the Prior Engine.'
        assert self.variables is not None, \
            'There are no variables for prior retrieval specified/passed on.'

    def get_priors(self):
        """
        Get prior data.
        calls *_get_prior* for all priors in config.

        :returns: dictionary with prior names/(state vector,
                  inverse covariance matrix) as key/value
        :rtype: dictionary

        """
        res = {}
        for var in self.variables:
            if var is not 'General':
                res.update({var: self._get_prior(var)})
        # return self._concat_priors(res)
        return res

    def _get_config(self):
        """
        Load config from self.configfile.
        writes to self.config.

        :returns: nothing
        """
        with open(self.configfile, 'r') as cfg:
            self.config = yaml.load(cfg)
        assert self.config['Prior'] is not None, \
            ('There is no prior config information in {}'
             .format(self.configfile))

    def _get_prior(self, var):
        """ Called by get_priors for all variables to be inferred.\
        For specific variable/prior (e.g. sm climatology) get prior\
        info and calculate/provide prior.

        :param var: prior name (e.g. sm, lai, ..)
        :returns: -
        :rtype: -

        """
        # TODO ad correct sub routines from Joris
        # subengine dictionary contains var: subroutine
        subengine = {
            'sm': SoilMoisturePrior,
            'dielectric_const': '',
            'roughness': RoughnessPrior,
            'lai': VegetationPrior,
            'cab': VegetationPrior
        }
        var_res = {}
        assert var in self.config['Prior'].keys(), \
            'Variable to be inferred not in config.'
        assert var in subengine,\
            ('No sub-enginge defined for variable to be inferred ({}).'
             .format(var))
        print('for variable *{}* getting'.format(var))

        # test if prior type is specified (else return empty dict):
        try:
            self.config['Prior'][var].keys() is not None
        except AttributeError as e:
            print('[WARNING] No prior type for {} moisture prior specified!'
                  .format(var))
            return

        # fill variable specific dictionary with all priors (clim, recent, ..)
        # TODO concatenation necessary?
        for ptype in self.config['Prior'][var].keys():
            # pass conig and prior type to subclass/engine
            try:
                prior = subengine[var](ptype=ptype, config=self.config,
                                       datestr=self.datestr, var=var)
                var_res.update({ptype: prior.RetrievePrior()})
                print('  '+ptype)
            except AssertionError as e:
                print('[WARNING] Sub-engine for *{}* {} prior not implemented!'
                      .format(ptype, var))
                # print(e)
        print('prior files.')
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
    Return state vector and inverse covariance matrix for priors.

    :param date: The date (time?) for which the prior needs to be derived
    :param variables: A list of variables (sm, lai, roughness, ..)
    for which priors need to be available

    :return: dictionary with keys being the variables and
    values being tuples of filenames and bands
    """

    return (PriorEngine(datestr=datestr, variables=variables,
                        config=config)
            .get_priors())


if __name__ == '__main__':
    print(get_mean_state_vector(datestr="2017-03-01", variables=['sm','lai', 'cab']))
