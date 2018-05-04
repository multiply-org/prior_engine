#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Integrate user defined priors to MULTIPLY engine.

    Copyright (C) 2018  Thomas Ramsauer
"""

import argparse
import datetime
import os
import tempfile

import yaml

from .prior import Prior
from .prior_engine import _get_config

__author__ = "Thomas Ramsauer"
__copyright__ = "Thomas Ramsauer"
__license__ = "gpl3"
__version__ = "0.4.0"
__maintainer__ = "Thomas Ramsauer"
__email__ = "t.ramsauer@iggf.geo.uni-muenchen.de"
__status__ = "prototype"


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
        self.configfile = kwargs.get('config', None)
        if self.configfile is None:
            self.configfile = kwargs.get('configfile', None)
        self.config = _get_config(self.configfile)

        # TODO read variables from config? are all possibly to infer variables
        # included? or is it better to have a defined list somewhere
        # (e.g. as class variable in PriorEngine like subengines)?
        self.variables = [k for k in self.config['Prior'].keys()
                          if k != 'General']

    def check_user_config(self):
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

    def _count_defined_userpriors(self):
        """Checks if a user defined prior is already defined in config, to:
        Writing user prior config:
           get count so that the next section can be written (user1, user2,...)

        :returns: count of user defined prior sections.
        :rtype: int

        """
        existing_userprior = 0
        for ptype in self.config.keys():
            if 'user' in ptype:
                existing_userprior += 1
        return existing_userprior

    def _write_userconf(self):
        """write gathered user input to configfile.

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
                {'dir': self.prior_directory}
                # {''}
             })
        # write self.config to new configfile
        with open(self.configfile, 'w') as cfg:
            cfg.write(yaml.dump(self.config, default_flow_style=False))

    def define_prior(self):
        """CLI to write config of user defined prior.

        :returns: configfile name (with path)
        :rtype: string

        """

        parser = argparse.ArgumentParser(
            description=('Utility to integrate User Prior data in MULTIPLY'),
            prog="user_prior.py",
            # usage='%(prog)s directory [-h] [-p]'
            )

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
                                  'config.\nIf None, a temporary file location '
                                  'will be used.'))
        parser.add_argument('-fn', '--filename', type=str, metavar='',
                            required=False,
                            action='store', dest='filename',
                            help=('Filename of new user '
                                  'config. Only has effect if path_to_config'
                                  ' is specified.\nIf None, a temporary '
                                  'filename will be used.'))

        args = parser.parse_args()

        if args.filename is not None and args.path_to_config is None:
            print('\n[WARNING] Entered config file name ({}) will be omitted '
                  '--> no path specified (-c/--path_to_config)!\n'
                  .format(args.filename))
        def _check_path(path):
            try:
                assert os.path.isdir(path), \
                  ('Entered path ({}) does not exist!'.format(path))
                print('Entered valid path (\'{}\').'.format(path))
                return path
            except AssertionError as e:
                # TODO creation of new folder?
                parser.error(e)

        # check prior data directory
        if args.prior_directory is not None:
            self.prior_directory = _check_path(args.prior_directory)

        # check config directory
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.path_to_config is not None:
            path_to_config = _check_path(args.path_to_config)
        else:
            # create temporary files to write config to:
            temp_conf = tempfile.NamedTemporaryFile(
                prefix='PriorEngine_config_{}_'.format(now), suffix='.yml')
            path_to_config = temp_conf.name

        # if valid path entered but missing filename:
        if not os.path.isfile(path_to_config):
            if args.filename is not None:
                self.configfile = os.path.join(path_to_config, args.filename)
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
                    print('\n [WARNING] {}\n'.format(e))
        else:
            self.configfile = path_to_config
        print('User config file: {}'.format(self.configfile))

        self.variable = args.prior_variable
        self._write_userconf()
        # TODO update self.configname??
        # self.configfile = temp_conf.name 
        # TODO log temp file name, add logger anyways
        return self.configfile


def main():
    # with open("./sample_config_prior.yml", 'r') as cfg:
    #     config = yaml.load(cfg)
    # U = UserPrior(ptype='user1', config=config,
    #               datestr="2017-03-01", var='sm')
    U = UserPriorInput(configfile="./sample_config_prior.yml")
    U.define_prior()

# def get_mean_state_vector(datestr: str, variables: list,
#                           ) -> dict:
#     """
#     Return dictionary with variable dependent sub dictionary with prior type
#     (key) and filenames of prior files (values).

#     :param datestr: The date (time?) for which the prior needs to be derived
#     :param variables: A list of variables (sm, lai, roughness, ..)
#     for which priors need to be available

#     :return: dictionary with keys being the variables and
#     values being a dictionary of prior type and filename of prior file.
#     """

#     return (PriorEngine(datestr=datestr, variables=variables,
#                         config=config)
#             .get_priors())

#                 prior = self.subengine[var](ptype=ptype, config=self.config,
#                                             datestr=self.datestr, var=var)


if __name__ == '__main__':
    main()
