#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Convert SMAP L4 HDF5 files to daily tiff files to be used by MULTIPLY

    Copyright (C) 2018  Thomas Ramsauer
"""

import glob
import os
import tempfile
import subprocess
import argparse

import h5py
from dateutil.parser import parse
from osgeo import gdal


__author__ = "Thomas Ramsauer"
__copyright__ = "Thomas Ramsauer"
__license__ = "gpl3"
__version__ = "0.0.1"
__maintainer__ = "Thomas Ramsauer"
__email__ = "t.ramsauer@iggf.geo.uni-muenchen.de"
__status__ = "development"


def create_SMAP_GeoTiff(filename):
    """
    create geotiffs from hdf5 layers in EASE grid 2.0 (EPSG:6933)
    Create stacked vrt files containing data and std
    """
    print('creating GeoTiff for {}.'.format(filename))
    data = ('HDF5:"{}"://Analysis_Data/sm_surface_analysis'
            .format(filename))
    std = ('HDF5:"{}"://Analysis_Data/sm_surface_analysis_ensstd'
           .format(filename))
    # get date info for vrt filename
    date, datestr = _get_date_from_SMAP(filename)

    data_out = '{}_sm.tif'.format(filename)
    std_out = '{}_std.tif'.format(filename)

    # create geotiffs from hdf5 layers

    # GDAL Python API not working as expected...:
    # gdal.Translate(destName=data_out, srcDS=data,
    #                outputSRS=srs, outputBounds=bounds)
    # gdal.Translate(destName=std_out, srcDS=std)

    # for EPSG:6933: EASE grid 2.0
    # out_srs = "+proj=cea +lon_0=0 +lat_ts=30 +ellps=WGS84 +units=m"
    # out_bounds = "-17367530.45 7314540.11 17367530.45 -7314540.11"

    subprocess.run('gdal_translate -a_ullr -17367530.45 7314540.11 17367530.45'
                   ' -7314540.11 -a_srs "+proj=cea +lon_0=0 +lat_ts=30'
                   ' +ellps=WGS84 +units=m" -a_nodata -9999 {} {}'
                   .format(data, data_out),
                   shell=True, check=True)
    subprocess.run('gdal_translate -a_ullr -17367530.45 7314540.11 17367530.45'
                   ' -7314540.11 -a_srs "+proj=cea +lon_0=0 +lat_ts=30'
                   ' +ellps=WGS84 +units=m" -a_nodata -9999 {} {}'
                   .format(std, std_out),
                   shell=True, check=True)

    # Create stacked vrt files:
    out_fn = '{}/SMAP_{}.vrt'.format(filename.rsplit('/', 1)[0], datestr)
    gdal.BuildVRT(out_fn, [data_out, std_out], separate=True)
    out_fn2 = 'vrt/SMAP_{}.vrt'.format(datestr)
    gdal.BuildVRT(out_fn2, [data_out, std_out], separate=True)


def _create_HDF5_filelist(path):
    """
    get all HDF5 files in path
    """
    files = sorted(glob.glob('**/*h5', recursive=True))
    if len(files) > 0:
        return files
    else:
        print('no files found in {}!'.format(path))


def _get_variable_from(filename, group, variable):
    """
    get data for variable from group HDF5 file (filename)
    """
    with h5py.File(filename, 'r') as f:
        dset = f['{}/{}'.format(group, variable)]
        # assert dset.shape == (3600, 1800), ('{}: incompatible shape ({})'
        #                                     .format(filename, dset.shape))
        return dset[:]


def _get_date_from_SMAP(filename):
    """
    extract date from file name
    """
    d = filename.split('/')[-1].split('_')[4]
    t = parse(d)
    return t, d


# def create_vrt_for_timestep(path):
#     """
#     OBSOLETE! Done in create geotiffs already
#     Go in daily folders and create daily file of the 3-hourly files.
#     """
#     for dirname, dirnames, filenames in os.walk(path):
#         # print path to all subdirectories first.
#         os.chdir(dirname)
#         for directory in dirnames:
#             os.chdir(directory)
#             print('change dir to {}'.format(directory))

#             files_mean = sorted(glob.glob('*sm.tif'))
#             files_std = sorted(glob.glob('*std.tif'))
#             assert len(files_mean) == len(files_std)
#             print(files_mean, files_std)

#             date, t = get_date_from_SMAP(create_HDF5_filelist('.')[0])
#             print('Date: ' + t[:8])
#             for i, f_sm in enumerate(files_mean):
#                 print(i)
#                 print(f_sm[:30])
#                 print(files_std[i][:30])
#                 assert f_sm[:30] == files_std[i][:30]
#                 gdal.BuildVRT('SMAP_'+t+'stacked.vrt', [f_sm, files_std[i]],
#                               separate=True)

#             os.chdir(dirname)
#             print('change dir to {}'.format(dirname))
#             break


def create_daily_tif(path):
    """Create daily SMAP tif with means of sm / unc in layer 1 & 2.

    :param path: path to daily folders
    :returns: -
    :rtype: -

    """
    for dirname, dirnames, filenames in os.walk(path):
        # print path to all subdirectories first.
        os.chdir(dirname)
        print('Entered \'{}\'.'.format(dirname))
        for directory in dirnames:
            out_fn, out_vrt = None, None
            curdir = os.path.join(dirname, directory)
            os.chdir(curdir)
            print('Entered \'{}\'.'.format(os.getcwd()))

            files = sorted(glob.glob('*vrt'))
            try:
                assert len(files) == 8, \
                    ("Directory ({}) did not contain {} vrts."
                     .format(directory, len(files)))
            except AssertionError as e:
                # TODO replace print statements with logger
                with open('../SMAP.log', 'a') as the_file:
                    the_file.write(str(e))
                os.chdir(dirname)
                print('change dir back to {}'.format(os.getcwd()))
                continue

            date, t = _get_date_from_SMAP(
                sorted(_create_HDF5_filelist(os.getcwd()))[0])

            out_fn = "SMAP_daily_" + t[:8]

            # TODO make test more robust with Regex:
            if os.path.exists(out_fn+'.tif'):
                continue
            print('\n\n{}\n    ---->>\n{}\n\n'.format(files, out_fn))
            abc = [chr(i) for i in range(ord('A'), ord('Z')+1)]
            mean_instr, unc_instr, calc_instr = '', '', ''
            # create input strings for gdal calculate call
            for i, f in enumerate(files):
                mean_instr += ('-{abc} {fn} --{abc}=1 '
                               .format(abc=abc[i], fn=f))
                unc_instr += ('-{abc} {fn} --{abc}=2 '
                              .format(abc=abc[i], fn=f))
            calc_instr = '+'.join(map(str, abc[:i+1]))
            # create temporary files to write mean mean&unc to
            mean_tf = tempfile.NamedTemporaryFile(suffix='_mean.tif')
            unc_tf = tempfile.NamedTemporaryFile(suffix='_unc.tif')

            # create means of input file mean and uncertainty files
            # TODO replace subprocess call wth module (or rasterio)?
            subprocess.run('gdal_calc.py {} --outfile={} --overwrite '
                           '--calc="({})/{}"'
                           .format(mean_instr, mean_tf.name, calc_instr,
                                   str(len(files))),
                           shell=True, check=True)
            subprocess.run('gdal_calc.py {} --outfile={} --overwrite '
                           '--calc="({})/{}"'
                           .format(unc_instr, unc_tf.name, calc_instr,
                                   str(len(files))),
                           shell=True, check=True)

            # http://gdal.org/python/osgeo.gdal-module.html#TranslateOptions
            out_vrt = gdal.BuildVRT(out_fn+'.vrt', [mean_tf.name, unc_tf.name],
                                    separate=True)
            gdal.Translate(out_fn+'.tif', out_vrt)
            print('created {}.tif in {}'.format(out_fn, os.getcwd()))
            # close/delete temporary files
            mean_tf.close()
            unc_tf.close()
            os.chdir(dirname)


def main():
    parser = argparse.ArgumentParser(
        description=('Utility to create daily SMAP '
                     'pior files to be used with the MULTIPLY Prior Engine.'))
    parser.add_argument('path', type=str,
                        help=('Path to the folder containing SMAP HDF5 files.'
                              '\ne.g. SMAP/n5eil01u.ecs.nsidc.org/DP4/SMAP/'
                              'SPL4SMAU.003'))

    args = parser.parse_args()
    if not os.path.isdir(os.path.expanduser(args.path)):
        raise FileNotFoundError(f"\"{args.path}\" does not exist!")

    os.chdir(args.path)
    fl = _create_HDF5_filelist(args.path)
    i = 1
    for f in fl:
        print('\n\nProcessing timestep {}/{}\n'.format(i, len(fl)))
        create_SMAP_GeoTiff(f)
        i += 1

    create_daily_tif(args.path)


if __name__ == '__main__':
    main()
