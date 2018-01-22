#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 T Ramsauer. All rights reserved.

"""
    Write GeoTiffs from climatology NetCDF files.

    Copyright (C) 2017  Thomas Ramsauer
"""

import os
import gdal
import numpy as np
from netCDF4 import Dataset


__author__ = "Joris Timmermans, Thomas Ramsauer"
__copyright__ = "Joris Timmermans, Thomas Ramsauer"
__license__ = "gpl3"


def WriteGeoTiff_from_climNetCDF(filename, varname,
                                 lyr_mean='mean', lyr_unc='unc',
                                 new_no_data_value=None,
                                 upper_no_data_thres=None,
                                 lower_no_data_thres=None):
    """Write GeoTiffs from NetCDF

    :param filename: 
    :param varname: 
    :param lyr_mean: 
    :param lyr_unc: 
    :returns: 
    :rtype: 

    """
    d = Dataset(filename, 'r')
    lons_in = d.variables['lon'][:]
    lats_in = d.variables['lat'][:]
    # print(d.variables)
    Nlayers = 2
    # latstr = '[{02.0f} {02.0f}N'.format(lats_in[0], lat_study[1])
    # lonstr = '[{03.0f} {03.0f}E]'.format(lon_study[0], lon_study[1])

    drv = gdal.GetDriverByName("GTIFF")
    for month in np.arange(0, 12, 1):
        print('month: ' + str(month+1))
        fn_out = filename.split('.')[0] + '_{:02d}'.format(month+1) + '.tiff'
        out_shape = (d.variables[lyr_mean][month].shape)
        # print(fn_out, out_shape[0], out_shape[1],Nlayers)
        dst_ds = drv.Create(fn_out, out_shape[1], out_shape[0],
                            Nlayers, gdal.GDT_Float32,
                            options=["COMPRESS=LZW",
                                    "INTERLEAVE=BAND",
                                    "TILED=YES"])
        resx = lons_in[0][1] - lons_in[0][0]
        resy = lats_in[1][0] - lats_in[0][0]
        dst_ds.SetGeoTransform([
             np.min(lons_in), resx, 0,
             np.max(lats_in), 0, -np.abs(resy)])
        dst_ds.SetProjection(
            'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS84",'
            '6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
            'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]')

        means = d.variables[lyr_mean][month][:]
        unc = d.variables[lyr_unc][month][:] 
        if new_no_data_value is not None:
            if upper_no_data_thres is not None:
                means[means >= upper_no_data_thres] = new_no_data_value
                unc[unc >= upper_no_data_thres] = new_no_data_value
            if lower_no_data_thres is not None:
                means[means <= lower_no_data_thres] = new_no_data_value
                unc[unc <= lower_no_data_thres] = new_no_data_value
        dst_ds.GetRasterBand(1).WriteArray(means[::-1])
        dst_ds.GetRasterBand(1).SetDescription(varname + '-mean')
        dst_ds.GetRasterBand(2).WriteArray(unc[::-1])
        dst_ds.GetRasterBand(2).SetDescription(varname + '-unc')
        dst_ds = None


if __name__ == '__main__':
    os.chdir('/home/thomas/Code/prior-engine/aux_data')
    WriteGeoTiff_from_climNetCDF(filename=('CCI_SM_climatology_eur_merged_inv'
                                 '.nc'), varname='sm', lyr_mean='sm',
                                 lyr_unc='sm_stdev', new_no_data_value=-999,
                                 upper_no_data_thres=10)
