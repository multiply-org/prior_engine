#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Prior Engine for MULTIPLY.

    Copyright (C) 2018  Thomas Ramsauer
"""


import datetime
from dateutil.parser import parse
import numpy as np
import yaml

if __name__ == '__main__':
    from soilmoisture_prior import RoughnessPrior, SoilMoisturePrior
    # from vegetation_prior import vegetation_prior

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
        self.date = kwargs.get('date', None)
        self.variables = kwargs.get('variables', None)
        # self.priors = self.config['Prior']['priors']

        self._get_config()
        self._check()

    def _check(self):
        assert self.config is not None, \
            'Could not load configfile.'
        # assert self.priors is not None, \
        #     'There is no prior specified in configfile.'
        assert self.date is not None, \
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
            # 'lai': vegetation_prior
        }
        var_res = {}
        assert var in self.config['Prior'].keys(), \
            'Variable to be inferred not in config.'
        assert var in subengine,\
            'Variable to be inferred not in config.'
        print('for var *{}* getting'.format(var))

        # fill variable specific dictionary with all priors (clim, recent, ..)
        # TODO concatenation necessary?
        for ptype in self.config['Prior'][var].keys():
            assert ptype is not None, \
                'No prior type for soil moisture prior specified!'
            # pass conig and prior type to subclass/engine
            try:
                prior = subengine[var](ptype=ptype, config=self.config,
                                       date=self.date)
                var_res.update({ptype: prior.initialize()})
                print('  '+ptype)
            except AssertionError as e:
                print('[WARNING] Sub-engine for *{}* {} prior not implemented!'
                      .format(ptype, var))
                # print(e)
        print('prior.')
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


class Prior(object):
    def __init__(self, **kwargs):
        self.ptype = kwargs.get('ptype', None)
        self.config = kwargs.get('config', None)
        self.date = kwargs.get('date', None)
        self._check()
        self._create_time_vector()
        self._create_month_id()

    def _check(self):
        assert self.ptype is not None, 'Invalid prior type'
        assert self.config is not None, 'No config available.'

    def _create_time_vector(self):
        """Creates a time vector dependent on start & end time and time interval
        from config file.
        A vector containing datetime objects is written to self.time_vector.
        A vector containing months ids (1-12) for each timestep is written to
        self.time_vector_months.

        :returns: -
        :rtype: -
        """
        # date_format = ('%Y-%m-%d')
        s = self.config['General']['start_time']
        e = self.config['General']['end_time']
        interval = self.config['General']['time_interval']
        t_span = (e-s).days + 1
        # print(t_span)

        # create time vector

        self.time_vector = [(s+(datetime.timedelta(int(x))))
                            for x in np.arange(0, t_span, interval)]

        # create list of month ids for every queried point in time:
        idt_months = [(s+(datetime.timedelta(int(x)))).month
                      for x in np.arange(0, t_span, interval)]
        self.time_vector_months = idt_months

    def _create_month_id(self):
        # assert parsing of self.date is working.
        assert type(parse(self.date)) is datetime.datetime,\
            'could not parse date {}'.format(self.date)
        # parse (dateutil) self.date to create datetime.datetime object
        self.date = parse(self.date)
        # get month id/number from self.date
        self.date_month_id = self.date.month

    def initialize(self):
        """Initialiszation routine. Should be implemented in child class.
        Prior calculation is initialized here.

        :returns: -
        :rtype: -

        """
        assert False, 'Should be implemented in child class'


def get_mean_state_vector(date: str, variables: list,
                          config: str="./sample_config_prior.yml") -> dict:
    """
    Return state vector and inverse covariance matrix for priors.

    :param date: The date (time?) for which the prior needs to be derived
    :param variables: A list of variables (sm, lai, roughness, ..)
    for which priors need to be available

    :return: dictionary with keys being the variables and
    values being tuples of filenames and bands
    """

    return (PriorEngine(date=date, variables=variables,
                        config=config)
            .get_priors())


if __name__ == '__main__':
    print(get_mean_state_vector(date="2017-03-01", variables=['sm']))
