#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 T Ramsauer. All rights reserved.

"""
    Create ESA CCI SM climatology.

"""

import datetime
import argparse
import os
import glob
from joblib import Parallel, delayed
import numpy as np
import yaml
import write_geotiffs
import xarray as xr

__author__ = "Thomas Ramsauer"
__copyright__ = "Thomas Ramsauer"
__license__ = "gpl3"


def prepare_yearly_aggregation(path_to_folders):
    """Aggregates daily files to yearly Netcdf files and copies\
    them to ../yearly/

    :param path_to_folders: path to folders that contain daily ESA_CCI files.
    :returns: -
    :rtype: -

    """
    # years = range(1981, 1982)
    os.chdir(path_to_folders)
    try:
        os.mkdir('./yearly/')
    except:
        print('Could not create ./yearly/ in {}.'.format(path_to_folders))

def yearly_aggregation(path_to_folders, year, cdo_in):
    """
    Aggregate one year with cdo.
    """
    try:
        os.chdir(path_to_folders + '/{}/'.format(year))
        print("CDO calculating:\n{}".format(cdo_in))
        os.system("cdo mergetime {}".format(cdo_in))
    except:
        print('[WARNING] Could not process year {}.'.format(year))


def parallelize_yearly_aggregation(path_to_folders, years):
    """
    Loop over years in parallelization
    Create argument instances to process
    """

    arg_instances = [(year, "*{}*.nc ../yearly/ESA_CCI_SM_{}.nc"
                      .format(year, year)) for year in years]
    for i in (arg_instances):
        print(*i)
    # Perform parallel computation
    Parallel(n_jobs=-2, verbose=100, backend="multiprocessing")(
        delayed(yearly_aggregation)(path_to_folders, year, cdo_in) for year, cdo_in in arg_instances)


def extract_sm(files):
    """Extract sm variable from all yearly ESA_CCI files ('*[0-9]*.nc).

    :param path_to_folders: path to folders that contain daily ESA_CCI files.
    :returns: -
    :rtype: -

    """
    assert type(files) == list
    for f in files:
        fn_out = f.split(".")[0] + '_sm.nc'
        print(f, ' --> ', fn_out)
        os.system('ncks -v sm {} {}'.format(f, fn_out))


def parallelize_extract_sm(files):
    Parallel(n_jobs=-2, verbose=100, backend="multiprocessing")(
        delayed(extract_sm)([f]) for f in files)


def aggregate_all(path_to_folder):
    """Aggregates all 'ESA_CCI_SM*' files in folder via cdo mergetime.

    :param path_to_folder: folder containing files to aggregate.
    :returns: -
    :rtype: -

    """
    os.chdir(path_to_folder)
    os.system('cdo mergetime ESA_CCI_SM* ESA_CCI_SM.nc')


def create_monmeans(filename):
    fn_out = filename.rsplit('.')[0] + '_monmean.nc'
    os.system('cdo monmean {} {}'.format(filename, fn_out))


def calc_climatology(**kwargs):
    """Create a netCDF file with coordinates "month" (1-12) and "lat", "lon"
    from ESA CCI data.

    Keywords:
    ---------
    filename: uses this filename
    pattern: if no filename is specified, takes a 'pattern' (linux style) to
             search for file. defaults to "*mon*".

    :returns: -
    :rtype: -

    """
    fn = kwargs.get("filename", None)
    if not fn:
        pattern = kwargs.get("pattern", None)
        if pattern:
            fn = glob.glob(pattern)[0]
        else:
            fn = glob.glob("*mon*")[0]
    ds = xr.open_dataset(fn)
    clim = xr.Dataset(coords={"month": range(1, 13),
                              "lon":  ds.coords["lon"].data,
                              "lat":  ds.coords["lat"].data},
                      data_vars={"sm_mean": (('month', 'lat','lon'),
                                             ds.sm[:12].data*np.nan),
                                 "sm_std":  (('month', 'lat','lon'),
                                             ds.sm[:12].data*np.nan)})

    for i in range(12):
        clim.sm_mean[i, :, :] = ds.sm[i::12, :, :].mean(dim="time")
        clim.sm_std[i, :, :] = ds.sm[i::12, :, :].std(dim="time")

    clim.to_netcdf('ESA_CCI_SM_CLIM.nc')
    del ds, clim


def main():
    parser = argparse.ArgumentParser(
        description=('Utility to create ESA CCI soil moisture climatology '
                     'pior files to be used with the MULTIPLY Prior Engine'))
    parser.add_argument('config', type=str,
                        help=('Configuration file name.'))
    parser.add_argument('-r',
                        action='store_true', dest='rm_intermediate',
                        help=('Remove intermediate files?'))
    args = parser.parse_args()

    if args.rm_intermediate:
        raise NotImplementedError

    try:
        with open(args.config, 'r') as cfg:
            config = yaml.load(cfg)
    except FileNotFoundError as e:
        raise

    def _check_path(path):
        if not os.path.isdir(os.path.expanduser(path)):
            raise FileNotFoundError(f"\"{path}\" does not exist!")
    _check_path(config['path_to_yearly_folders'])
    _check_path(config['output_folder'])

    try:
        type(config['start'])
    except KeyError:
        config['start'] = 1980
    try:
        type(config['end'])
    except KeyError:
        # TODO add dynamic end (based on complete folder)
        config['end'] = datetime.datetime.today().date().year-1

    print("\nConfig:\n-------")
    for i, j in config.items():
        print(f"{i}: {j}")
    print()

    # TODO add logger instead of print arguments

    # set path to extracted yearly folders containing the
    # daily ESA CCI SM datasets:
    # -----------------------------------------
    path_to_folders = config['path_to_yearly_folders']

    # Aggregate daily files on a yearly basis
    # -----------------------------------------
    # prepare_yearly_aggregation(path_to_folders)
    # parallelize_yearly_aggregation(path_to_folders, range(config['start'],
    #                                                       config['end']+1))

    # Extract 'sm' variable from data sets to reduce size:
    # -----------------------------------------------------
    # os.chdir(path_to_folders + 'yearly/')
    # files = sorted(glob.glob('ESA_CCI_SM_*[0-9]*.nc'))
    # parallelize_extract_sm(files)

    # Move sm files to dedicated foder:
    # -----------------------------------------
    # print('Moving new files to \'./sm\'')
    os.chdir(path_to_folders + 'yearly/')
    # TODO not OS independent
    # os.system('mkdir sm')
    # os.system('mv *_sm.nc sm/')
    os.chdir('sm/')

    # Aggregate whole time series
    # -----------------------------------------
    # print('Aggregating all soil moisture files..')
    # aggregate_all('./')

    # create monthly means
    # -----------------------------------------
    # filename = ('ESA_CCI_SM.nc')
    # print('Starting to get monthly means for {}'.format(filename))
    # create_monmeans(filename)

    # Calculate climatology:
    # -----------------------------------------
    # fn = ('ESA_CCI_SM_monmean.nc')
    # print('Starting to create climatology for {}...'.format(fn), end="")
    # calc_climatology(filename=fn)
    # print("Done!")

    # Create GeoTiffs
    # ----------------------
    print('Starting to weite GeoTiffs from generated climaology.')
    write_geotiffs.WriteGeoTiff_from_climNetCDF('ESA_CCI_SM_CLIM.nc',
                                                out_dir=config['output_folder'],
                                                varname='sm',
                                                lyr_mean='sm_mean',
                                                lyr_unc='sm_std',
                                                new_no_data_value=-999,
                                                upper_no_data_thres=1000,
                                                flip_lat=False)


if __name__ == '__main__':
    main()
