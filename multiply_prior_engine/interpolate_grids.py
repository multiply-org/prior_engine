#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 T Ramsauer. All rights reserved.

"""
interpolate grids between dates
"""
import datetime
import sys
from dateutil.parser import parse
import numpy as np
# import pickle
import gdal
import re
import os
from matplotlib import pyplot as plt

__author__ = "Thomas Ramsauer"
__copyright__ = "Thomas Ramsauer"
__license__ = "gpl3"


def get_files(path, pattern=r".*[0-9]{8}.*.tif*"):
    """
    Get all files with certain pattern in filename from path
    """

    fl = []
    for root, dirs, files in os.walk(path):
        for basename in files:
            if re.match(r'{}'.format(pattern), basename):
                fl.append(os.path.join(os.path.abspath(root), basename))

    return sorted(fl)


def get_date_from_file_name(fn):
    """
    parse (dateutil) datestr to create datetime.datetime object
    """

    date = parse(re.findall(r'\d{8}', os.path.basename(fn))[0]).date()
    return date

#     try:
#         date = parse(self.datestr)
#     except TypeError as e:
#         print('[WARNING] No time info was passed to Prior Class!')
#         return
#     # assert parsing of self.date is working.
#     assert type(date) is datetime.datetime,\
#         'could not parse date {}'.format(date)

#     # get month id/number from self.datestr
#     # self.date_month_id = self.date.month
#     self.date8 = int(str(self.date.date()).replace('-', ''))
#     return date


def get_timespan(fl, interval=1):
    """Return length of datelist and actual list with dates as tuple

    :param fl: file list
    :param interval: interval (days) for creation of date list

    >>> get_timespan(['testfile_20190101.tiff', 'testfile_20190103'])
    (3,
     [datetime.date(2019, 1, 1),
      datetime.date(2019, 1, 2),
      datetime.date(2019, 1, 3)])
    """

    s = get_date_from_file_name(sorted(fl)[0])
    e = get_date_from_file_name(sorted(fl)[-1])
    t_span = (e - s).days + 1

    # create list of day ids for every queried point in time:
    dates = [(s + (datetime.timedelta(int(x))))
             for x in np.arange(0, t_span, interval)]
    return t_span, dates


def get_band_as_array(fn, band=None, fillvalue=-999.):
    """Get band from GeoTiff with fillvalues replaced by np.nan

    :param fn: file name
    :param band: band in tiff file (specify if multiple available, else: None)
    :param fillvalue:  fill value to be replaced by 'np.nan'

    """
    ds = gdal.Open(fn)
    dims = len(ds.ReadAsArray().shape)

    if (dims < 3 and band is not None):
        print(f"Only found {dims} dimensions in raster {fn}.")
        band = None
    if band is not None:
        dsmatrix = ds.ReadAsArray(
            xoff=0, yoff=0,
            xsize=ds.RasterXSize, ysize=ds.RasterYSize)[band]
    else:
        dsmatrix = ds.ReadAsArray(
            xoff=0, yoff=0,
            xsize=ds.RasterXSize, ysize=ds.RasterYSize)

    # replace fillvalue with numpy nan
    if fillvalue:
        inds = np.where(dsmatrix == fillvalue)
        dsmatrix[inds] = np.nan

    return dsmatrix


def create_filled_stack(fl, band, fillvalue):

    # get dates and timespan (daily) from  filenames in filelist
    (t_span, dates) = get_timespan(fl)

    # get spatial dimensions from first file in list
    dims = len(gdal.Open(fl[0]).ReadAsArray().shape)
    assert dims > 1
    if dims == 3:
        _, idx, idy = gdal.Open(fl[0]).ReadAsArray().shape
    elif dims == 2:
        idx, idy = gdal.Open(fl[0]).ReadAsArray().shape

    # create nd-array with final dimensions
    stack = np.ndarray(shape=(t_span, idx, idy),
                       dtype=float) * np.nan

    # write bands from files into stack at appropriate index based on date
    for f in fl:
        date = get_date_from_file_name(f)
        idd = [id for (id, d) in enumerate(dates) if d == date]
        assert len(idd) == 1
        stack[idd[0], :, :] = get_band_as_array(f, band, fillvalue)

    assert len(stack.shape) == 3
    print("Created filled stack (with NaNs).")

    return stack, dates


def interpolate_stack(fl, band, fillvalue, **kwargs):
    """
    interpolate a filled stack
    OR if keyword argument 'single_date' is given:
    return single grid with distance aware interpolated values from the closest
    grids before and after

    :param fl: filelist
    :param band: band of geotiff
    :param fillvalue: fillvalue in girds to be interpolated

    """
    single_date = kwargs.get('single_date', None)
    p, dates = create_filled_stack(fl, band, fillvalue)

    if not single_date:
        # TEMPORARY CHECK:  # ####
        # plt.pcolormesh(p[0, ::-1, :])
        # plt.colorbar()
        # plt.show()
        # ########################

        # 1st solution:
        # ----------------
        filled_stack = np.apply_along_axis(pad, 0, p)
        return filled_stack, dates

        # multiprocessing
        # ----------------

        # create a list of 1d numpy arrays to use as input to pool.map()
        # There's perhaps a better way but this list comprehension suffices
        # print("Converting stack to list for multiprocessing...", end="")
        # ins = [p[:, idx, idy]
        #        for idx in range(p.shape[1])
        #        for idy in range(p.shape[2])]
        # print("done.\n")

        # use shared array to write to
        # res = np.ctypeslib.as_ctypes(p)
        # shared_array = multiprocessing.sharedctypes.RawArray(res._type_, res)

        # print("Starting multiprocessing:\n")
        # # pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
        # # outs = pool.map(pad, ins)
        # with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
        #     outs = list(tqdm.tqdm(pool.imap(pad, ins), total=len(ins)))
        # print('DONE')

        # TODO needs to be in if __name__ == main?? due to tqdm?
        # check out if pickle dump is really empty and full if in if name main

        # shared, can be used from multiple processes
        # mp_arr = mp.Array(c.c_double, n*m)
        # # then in each new process create a new numpy array using:
        # mp_arr and arr share the same memory
        # arr = np.frombuffer(mp_arr.get_obj())
        # # make it two-dimensional
        # b = arr.reshape((n,m))

        # pool.close()
        # pool.join()

        # print(len(outs))
        # print(outs[0])

        # # convert back to array and set the correct shape
        # print("Reshaping output...", end="")
        # filled_stack = np.array(outs).reshape(p.shape)
        # print("done.")
        # print(len(outs))
        # return filled_stack

        # 4th solution - ravel
        # ---------------------
        # print("starting apply to axis")
        # print(f"done out of {p.shape[1]*p.shape[2]}:")
        # import pandas as pd
        # import dask.dataframe as dd
        # from dask.diagnostics import ProgressBar
        # from dask.multiprocessing import get
        # import swifter
        # length = p.shape[1]*p.shape[2]
        # # res = np.apply_along_axis(
        #             pad, 0,
        # #           p.reshape(p.shape[0], p.shape[1]*p.shape[2]),
        # #                     length=length)

        # pdf = pd.DataFrame(p.reshape(
        #     p.shape[0], p.shape[1]*p.shape[2]))
        # print(1)
        # print(type(pdf[0][0]))
        # pbar = ProgressBar()
        # pbar.register()
        # ddf = dd.from_pandas(pdf, npartitions=4*multiprocessing.cpu_count())
        # print(2)
        # dddata = ddf.map_partitions(
        #     lambda df: df.apply(
        #                    pad,
        #                    meta=('np.float64'), axis=0)).compute(get=get)
        # print("start computing")
        # res = dddata

        # # res = df.swifter.apply(pad, axis=0)  # , length=length)

        # return res.reshape(p.shape)

    else:
        # weighted interpolation of nearest doy grids:
        doy = to_doy(parse(single_date))
        dl = {}
        [dl.update({d: fn}) for (d, fn)
         in [(to_doy(get_date_from_file_name(f), f)) for f in fl]]

        before_doy = str(max([k for k in dl.keys() if int(k) < doy]))
        before_file = dl[before_doy]
        after_doy = str(min([k for k in dl.keys() if int(k) > doy]))
        after_file = dl[after_doy]
        before_weight = abs(doy - before_doy)
        after_weight = abs(doy - after_doy)

        return (
            ((get_band_as_array(before_file, band, fillvalue)*before_weight)
             + (get_band_as_array(after_file, band, fillvalue)*after_weight))
            / (before_weight+after_weight))


def to_doy(indate):
    return int(indate.date().strftime("%j"))


count = 0


def counter():
    global count
    count += 1
    yield count


def pad(data, **kwargs):
    """
    interpolate 1-D array with nans
    """
    length = kwargs.get('length', None)
    assert len(data.shape) == 1
    if length:
        sys.stdout.write("\r"+str(list(counter())[0]+100/int(length))+"%")
    else:
        sys.stdout.write("\r"+str(list(counter())[0]))
    sys.stdout.flush()
    good = np.isfinite(data)
    if max(good) == 0:
        return np.ones(good.shape) * np.nan
    interpolated = np.interp(np.arange(data.shape[0]),
                             np.flatnonzero(good),
                             data[good])
    return interpolated


def _iter_pad(data, idx):
    assert len(data.shape) == 3
    # print(f"iterating over x: {data.shape[1]}, y: {data.shape[2]}.")
    for idy in range(data.shape[2]):
        # print(f"\r Progress: "
        #       f"{i*100/(data.shape[1] * data.shape[2]):.1f} %",
        #       end="")
        # i += 1
        data[:, idx, idy] = pad(data[:, idx, idy])
    return data


def create_geotiffs_from_stack(stack, example_tiff):
    """
    write geotiffs from stack
    """
    raise NotImplementedError()
# #         # Get Geographic meta data
#         geo_trans_list = ds.GetGeoTransform()
#         proj_str = ds.GetProjection()
#         num_bands = ds.RasterCount

#         # Adapt to one bands or multi-bands
#         if num_bands > 1:
#             # Unfold array into pandas DataFrame
#             print(dsmatrix.shape)
#             rows = dsmatrix.shape[1]
#             cols = dsmatrix.shape[2]
    # geo = driver = gdal.GetDriverByName("GTiff")
    # outdata = driver.Create(outFileName, rows, cols, 1, gdal.GDT_UInt16)
    # sets same geotransform as input:
    # outdata.SetGeoTransform(ds.GetGeoTransform())
    # outdata.SetProjection(ds.GetProjection())##sets same projection as input
    # outdata.GetRasterBand(1).WriteArray(arr_out)
    # if you want these values transparent
    # outdata.GetRasterBand(1).SetNoDataValue(10000)
    # outdata.FlushCache() ##saves to disk!!
    # outdata = None
    # band=None
    # ds=None


def main():

    # calculate:
    # -------------
    path = "../aux_data/Climatology/SoilMoisture/"
    path = os.path.expanduser("~/Gome/")
    pattern = ".*[0-9]{8}.*.tif*"
    fl = get_files(path, pattern)
    for f in fl:
        print(f)

    # only specify if multiple bands are in tiff
    # (however, will be checked anyways)
    band = 0
    fillvalue = -999.
    stack, dates = interpolate_stack(fl, band, fillvalue)

    # pickle dump the array
    # ---------------------
    # out_fn = os.path.join(
    #     path, (f"filled_stack_"
    #            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"))
    # with open(out_fn, 'wb') as f:
    #     pickle.dump(stack, f)
    #     print(f"\nsaved stack to {out_fn}")
    # with open(out_fn+"_dates.txt", 'w') as df:
    #     for date in dates:
    #         df.write(date.strftime("%Y%m%d")+"\n")

    # load pickled array:
    # ---------------------
    # fn = sorted(glob.glob("*pkl"))[-1]
    # print(f"Opening {fn}")
    # with open(fn, "rb") as f:
    #     stack = pickle.load(f)

    plt.pcolormesh(stack[9, ::-1, :])
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
