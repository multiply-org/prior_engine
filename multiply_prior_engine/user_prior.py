#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Integrate user defined priors to MULTIPLY engine.

    Copyright (C) 2018  Thomas Ramsauer
"""

import argparse
import datetime
import os
import sys
import tempfile
import warnings
import pandas as pd

import yaml

from .prior import Prior
from .prior_engine import _get_config, default_config


__author__ = "Thomas Ramsauer"
__copyright__ = "Copyright 2018 Thomas Ramsauer"
__maintainer__ = "Thomas Ramsauer"
__email__ = "t.ramsauer@iggf.geo.uni-muenchen.de"


class UserPrior(Prior):
    """

    """

    def __init__(self, **kwargs):
        super(UserPrior, self).__init__(**kwargs)

    def RetrievePrior(self):
        """
        Initialize prior specific (climatological, ...) calculation.

        :returns: 
        
        """

    def userprior_conversion(self):
        """Convert user defined data for compatibility?

        :returns: 
        :rtype: 

        """
        pass


class UserPriorInput(object):

    def __init__(self, **kwargs):
        # config file so far only needed to verify that variables, which
        # prior information should be added for, are inferrable.
        self.configfile = None
        while self.configfile is None:
            self.configfile = kwargs.get('config', None)
            self.configfile = kwargs.get('configfile', None)
            # have a backup/default config:
            self.configfile = default_config

        assert self.configfile is not None, \
            ('No configuration filename passed to function. '
             'Add \'configfile=\' to method call.')
        self.config = _get_config(self.configfile)

        # TODO really read vars from config? are all possibly to infer vars
        # included? or is it better to have a defined list somewhere
        # (e.g. as class variable in PriorEngine like subengines)?
        self.variables = [k for k in self.config['Prior'].keys()
                          if k != 'General']
        # TODO instead read from prior engine?
        # self.variables = ['sm', 'cab', 'lai']
        assert self.variables is not None, \
            'UserPriorInput does not know about possibly to infer variables.'

    def check_for_user_config(self):
        """Checks subsection of user prior config for specific information
        and extracts them (e.g. directory).

        :returns: -
        :rtype: -

        """
        self.userconf = self.config['Prior'][self.variable][self.ptype]
        for k in self.userconf.keys():
            if 'dir' in k:
                self.dir = k
            if 'other_options' in k:
                pass

    def _check_path(self, path):
        try:
            assert os.path.isdir(path), \
                ('Entered path ({}) does not exist!'.format(path))
            return path
        except AssertionError as e:
            # TODO creation of new folder?
            try:
                parser.error(e)
            except:
                raise(e)

    def _count_defined_userpriors(self):
        """Checks if a user defined prior is already defined in config, to:
        Writing user prior config:
           get count so that the next section can be written (user1, user2,...)

        :returns: count of user defined prior sections.
        :rtype: int

        """
        # TODO add possibility for other names. these must be stored somewhere.
        existing_userprior = 0
        for ptype in self.config.keys():
            if 'user' in ptype:
                existing_userprior += 1
        return existing_userprior

    def _generate_userconf(self, configfile: str, new_configuration: dict):
        """ generate dictionary with user prior information.

        :param configfile: filename of configuration file to write
        :param: new_configuration: dictionary holding new prior information
        :returns: -
        :rtype: -

        """
        # configfile_name = kwargs.get('configfile_name', None)
        # if configfile_name is None:
        #     configfile_name = 'user_config.yml'

        # add information to config dictionary

        count = self._count_defined_userpriors()
        self.config['Prior'][self.variable].update(
            {'user{}'.format(count+1):
                new_configuration
             })

    def write_config(self, configuration, **kwargs):
        """Write configuration to a YAML file.

        :param configuration: configuration dictionary to write to file.
        :Keyword Arguments:
            * *path_to_config* (``str``) --
              path to config file. if None, a tempfile will be created.
            * *filename* (``str``) --
              Filename of new user config. Only has effect if path_to_config
              is specified.If None, a temporary filename will be used.

        :returns: config file name
        :rtype: string

        """
        path_to_config = kwargs.get('path_to_config', None)
        filename = kwargs.get('filename', None)

        if filename is not None and path_to_config is None:
            warnings.warn('Entered config file name ({}) will be omitted '
                          '--> no path specified!'.format(filename), Warning)

        # check config directory
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if path_to_config is not None:
            path_to_config = self._check_path(path_to_config)
        else:
            # create temporary files to write config to:
            temp_conf = tempfile.NamedTemporaryFile(
                prefix='PriorEngine_config_{}_'.format(now), suffix='.yml')
            path_to_config = temp_conf.name

        # if valid path entered but missing filename:
        if not os.path.isfile(path_to_config):
            if filename is not None:
                self.configfile = os.path.join(path_to_config, filename)
            else:
                self.configfile = os.path.join(
                    path_to_config, 'PriorEngine_config_{}.yml'.format(now))
            try:
                with open(self.configfile, "x") as f:
                    pass
                assert os.path.isfile(self.configfile)
            except FileExistsError as e:
                self.configfile = os.path.join(
                    path_to_config, "PriorEngine_config_{}.yml".format(now))
                with open(self.configfile, "x") as f:
                    warnings.warn(e, Warning)
                assert os.path.isfile(self.configfile)
        else:
            self.configfile = path_to_config
        print('User config file: {}'.format(self.configfile))

        with open(self.configfile, 'w') as cfg:
            cfg.write(yaml.dump(configuration, default_flow_style=False))
        return self.configfile

    def show_config(self, only_prior=True):
        """Display current prior configuration. Print to stdout and return.

        :returns: (prior engine) configuration
        :rtype: dictionary

        """
        if self.configfile is not None:
            if only_prior:
                print('MULTIPLY Prior Engine Configuration \n({}):\n\n{}'
                      .format(self.configfile, yaml.dump(self.config['Prior'],
                              default_flow_style=False)))
                return self.config['Prior']
            else:
                print('MULTIPLY Configuration \n({}):\n{}'
                      .format(self.configfile, yaml.dump(self.config,
                              default_flow_style=False)))
                return self.config
        else:
            print('MULTIPLY Configuration file has not been specified yet.'
                  ' Please specify \'configfile=\' when initializing class.')
            # sys.exit()

    def delete_prior(self, variable, ptype, write=True):
        """Delete / Unselect prior in config.

        :param variable: 
        :param ptype: 
        :param write: 
        :returns: 
        :rtype: 

        """
        try:
            removed = self.config['Prior'][variable].pop(ptype)
            print('Removed {} prior configuration.'.format(removed))
        except KeyError as e:
            warnings.warn('{}/{} not in configuration'
                          .format(variable, ptype), Warning)
        if write:
            self.write_config(self.config)

    def add_prior(self, prior_variable, **kwargs):
        """Adds directory, which holds user prior data, to config file.
        The user defined prior data sets have to be in the common form of gdal
        compatible files (e.g geotiff, vrt, ..). The `import_prior` utility may
        therefor be utilized. The new config will be written to \
        `path_to_config` or `filename`, please see `write_config`.

        :param prior_variable: variable, which the user prior data is \
                               supporting (e.g. lai, sm)

        :Keyword Arguments:
            * *path_to_config* (``str``) --
              path to config file. if None, a tempfile will be created.
            * *filename* (``str``) --
              Filename of new user config. Only has effect if path_to_config
              is specified.If None, a temporary filename will be used.
            * *prior_directory* (``str``) --
              Directory path where user prior data is stored.

        :returns: 
        :rtype: 

        """
        # config file specific info (default ones used if not present):
        path_to_config = kwargs.get('path_to_config', None)
        filename = kwargs.get('filename', None)

        # so far only directory as user defined configuration implemented
        # TODO needs more flexibility:
        prior_directory = kwargs.get('prior_directory', None)
        # TODO a date vector for the files has to be passed on or an utility
        # needs to be created to read the correct data for date (an utility to
        # find datestrings in filenames)
        
        # used in _generate_userconf for location in config file
        self.variable = prior_variable

        # check prior data directory
        if prior_directory is not None:
            self.prior_directory = self._check_path(prior_directory)

        # adding to new config
        nc = {}
        for arg in kwargs:
            if arg is not 'path_to_config' and\
               arg is not 'filename':
                nc.update({arg: kwargs[arg]})

        # generate new config dictionary with user info included
        self._generate_userconf(configfile=self.configfile,  # to read config
                                new_configuration=nc)  # updates self.config
        # write extended config to file
        self.write_config(path_to_config=path_to_config, filename=filename,
                          configuration=self.config)

    def import_prior(self, prior_variable, **kwargs):
        """Import user prior data in common MULTIPLY prior data format (gdal
        compatible file, 2 layers).
        Subroutines may be called.

        :param arg:
        :returns:
        :rtype:

        """
        # Check for files or dir of input data, and output directory; create
        # internal directory to store the converted data if not specified

        # config file specific info (default ones used if not present):
        path_to_config = kwargs.get('path_to_config', None)
        filename = kwargs.get('filename', None)

        # so far only directory as user defined configuration implemented
        # TODO needs more flexibility:
        prior_directory = kwargs.get('prior_directory', None)

        # used in _generate_userconf for location in config file
        self.variable = prior_variable

        # check prior data directory
        if prior_directory is not None:
            self.prior_directory = self._check_path(prior_directory)
        # -------------

        # Import data with suitable method:
        # TODO finish section below
        try:
            self.read_tabular()
        except ...:
            is_netcdf(data)
            self.read_netcdg()
            pd.read_csv()
        # ...

        def _read_tabular(data):
            d = pd.read_table(data)
            return d

        def _read_netcdf(data):
            # geoval? netCDF4? other? hdf5?
            pass

        # add prior to config
        try:
            self.add_prior(prior_variable=self.variable, ... )
            return 0
        except e:
            # log Error
            return 99  # ?


    def add_prior_cli(self):
        """CLI to include configuration for user defined prior.

        :returns: configfile name (with path)
        :rtype: string

        """

        parser = argparse.ArgumentParser(
            description=('Utility to integrate User Prior data in MULTIPLY'),
            prog="user_prior.py",
            # usage='%(prog)s directory [-h] [-p]'
            )

        # TODO add deletion of priors here? new required flags
        # for 'add', 'delete', 'show'

        parser.add_argument('-v', '--prior_variable', type=str, metavar='',
                            action='store', dest='prior_variable',
                            required=True, choices=self.variables,
                            help=('Variable to use the prior data for.\n'
                                  'Choices are: {}'.format(self.variables)))
        parser.add_argument('-d', '--prior_directory', type=str, metavar='',
                            action='store', dest='prior_directory',
                            required=True,
                            help=('Directory which holds specific user prior'
                                  ' data.'))

        parser.add_argument('-c', '--path_to_config', type=str, metavar='',
                            required=False,
                            action='store', dest='path_to_config',
                            help=('Directory of new user '
                                  'config.\nIf None, a temporary file location'
                                  ' will be used.'))
        parser.add_argument('-fn', '--filename', type=str, metavar='',
                            required=False,
                            action='store', dest='filename',
                            help=('Filename of new user '
                                  'config. Only has effect if path_to_config'
                                  ' is specified.\nIf None, a temporary '
                                  'filename will be used.'))

        args = parser.parse_args()
        self.add_prior(**vars(args))


def main():
    try:
        U = UserPriorInput(configfile="./sample_config_prior.yml")
        U.add_prior_cli()
    except ModuleNotFoundError as e:
        print(e)
        # run from outside module or install properly


if __name__ == '__main__':
    main()
