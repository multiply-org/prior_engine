#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Soil Moisture Prior Engine

    Copyright (C) 2017  Thomas Ramsauer
"""


import tempfile
# import datetime
import os
# import xarray as xr
import yaml
import numpy as np
import scipy as sp
# from scipy import spatial
# import re
import ast

from netCDF4 import Dataset


class PriorEngine(object):

    def __init__(self, **kwargs):
        self.configfile = kwargs.get('config', None)
        self._get_config()
        # self.priors = kwargs.get('priors', None)
        self.priors = self.config['Prior']['priors']
        # print(self.priors)
        self._check()

    def _check(self):
        assert self.config is not None, \
            'Could not load configfile.'
        assert self.priors is not None, \
            'There is no prior specified in configfile.'

    def get_priors(self):
        res = {}
        # for p in self.priors:
        for p in self.priors.keys():
            res.update({p: self._get_prior(p)})
        return res

    def _get_config(self):
        with open(self.configfile, 'r') as cfg:
            self.config = yaml.load(cfg)
        assert self.config['Prior'] is not None, \
            ('There is no prior config information in {}'
             .format(self.configfile))

    def _get_prior(self, p):
        if p[:2] == 'sm':
            assert self.priors[p]['type'] is not None, \
                'No prior type for soil moisture prior specified!'

            # pass config and prior type to subclass
            prior = SoilMoisturePrior(ptype=self.priors[p]['type'],
                                      config=self.config)
            # for ptype in self.priors.sm.type:
            #     try:
            #         prior = SoilMoisturePrior(ptype=self.priors[p][ptype])
            #     except:
            #         assert False, 'SoilMoisturePrior generation failed.'
        elif p == 'vegetation':
            assert False, \
                ('The veg prior should provide cross-correlated prior'
                 'information of vegetation characteristics')
        elif p == 'roughness':
            prior = RoughnessPrior(ptype=self.priors[p]['type'],
                                   lc_file=self.config.landcover,
                                   lut_file=self.config.luts['roughness'])
        else:
            assert False, 'Invalid prior'

        # calculate prior
        prior.calc()

        return prior.file  # return filename where prior file is located


class Prior(object):
    def __init__(self, **kwargs):
        self.ptype = kwargs.get('ptype', None)
        self.config = kwargs.get('config', None)
        self._check()

    def _check(self):
        assert self.ptype is not None, 'Invalid prior type'

    def calc(self):
        assert False, 'Should be implemented in child class'


class MapPrior(Prior):
    """
    prior which is based on a LC map
    and a LUT
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        lut_file : str
            filename of LUT file
        lc_file : str
            filename of landcover file
        """
        super(MapPrior, self).__init__(**kwargs)
        self.lut_file = kwargs.get('lut_file', None)
        assert self.lut_file is not None, 'LUT needs to be provided'

        self.lc_file = kwargs.get('lc_file', None)
        assert self.lc_file is not None, 'LC file needs to be provided'

        # check that files exist
        assert os.path.exists(self.lc_file)
        assert os.path.exists(self.lut_file)


class SoilMoisturePrior(Prior):
    """
    Soil moisture prior generation
    """
    def __init__(self, **kwargs):
        super(SoilMoisturePrior, self).__init__(**kwargs)

    def calc(self):
        if self.ptype == 'climatology':
            self.file = self._calc_climatological_prior()
        # elif self.ptype == 'recent':
            # self.file = self._get_recent_sm_proxy()
        else:
            assert False, '{} prior for sm not implemented'.format(self.ptype)

    def _get_climatology_file(self):
        """
        load pre-processed climatology into self.clim
        """
        assert (self.config['Prior']['priors']['sm_clim']
                           ['climatology_file']) is not None,\
            'There is no climatology file specified in the config!'

        # use xarray:
        # self.clim_data = xr.open_dataset(self.config['priors']['sm_clim']
        self.clim_data = Dataset(self.config['Prior']['priors']['sm_clim']
                                 ['climatology_file'])

        # fd, path = tempfile.mkstemp()
        # os.write(fd, 'tempfile')
        # os.close(fd)

    def _extract_climatology(self):
        """
        extract climatology values for ROI
        """
        clim = self.clim_data.variables['sm'][:]
        std = self.clim_data.variables['sm_stdev'][:]
        lats = self.clim_data.variables['lat'][:]  # extract/copy the data
        lons = self.clim_data.variables['lon'][:]

        # TODO wonky ROI loading
        ROI_ = (self.config['General'][0]['roi']
                .split(' ', 1)[1]
                .replace(' ', ''))
        ROI = ast.literal_eval(ROI_)

        # TODO insert check for ROI bounds:
        # if POI[0] < np.amin(lats) or POI[0] > np.amax(lats) or\
        #    POI[1] < np.amin(lons) or POI[1] > np.amax(lons):
        #     raise ValueError("POI's latitude and longitude out of bounds.")
        combined_LAT_LON = np.dstack([lats.ravel(), lons.ravel()])[0]
        mytree = sp.spatial.cKDTree(combined_LAT_LON)
        idx, idy = [], []
        for i in range(len(ROI)):
            dist, indexes = mytree.query(ROI[i])
            x, y = tuple(combined_LAT_LON[indexes])
            idx.append(indexes % clim.shape[2])
            idy.append(int(np.ceil(indexes / clim.shape[2])))

        # TODO check assignment
        # print(idx, idy)

        # extract sm data
        sm_area_ = clim[:, min(idy):max(idy)+1, :]
        sm_area = sm_area_[:, :, min(idx):max(idx)+1]

        # extract sm stddef data
        sm_area_std_ = std[:, min(idy):max(idy)+1, :]
        sm_area_std = sm_area_std_[:, :, min(idx):max(idx)+1]
        # sm_area_std = np.std(sm_area, axis=(1, 2))
        # sm_area_mean = np.mean(sm_area, axis=(1, 2))

        # TODO limit to months

        # date_format = ('%Y-%m-%d')
        # s = self.config['General'][1]['start_time']
        # e = self.config['General'][2]['end_time']
        # t_span = (e-s).days + 1
        # print(t_span)

        # print(sm_area)
        self.clim = sm_area
        self.std = sm_area_std

    def _calc_climatological_prior(self):
        self._get_climatology_file()
        self._extract_climatology()

        # create prior
        prior = np.array(range(0, 365))

        fd, path = tempfile.mkstemp()
        os.write(fd, prior)
        os.close(fd)

        # return prior.file
        self.file = path
        return path

    def _get_recent_sm_proxy(self):
        assert False, "recent sm proxy not implemented"


class RoughnessPrior(MapPrior):

    def __init__(self, **kwargs):
        super(RoughnessPrior, self).__init__(**kwargs)

    def calc(self):
        if self.ptype == 'climatology':
            self._read_lut()
            self._read_lc()
            self._map_lut()
            self.file = self.save()
        else:
            assert False

    def _read_lut(self):
        self.lut = 'abc'

    def _read_lc(self):
        self.lc = 'efg'

    def _map_lut(self):
        """
        should do the mapping of s, l, ACL type
        """
        self.roughness = self.lut + self.lc

    def save(self):
        """
        save mapped roughness data to file
        """
        return tempfile.mktemp(suffix='.nc')


P = PriorEngine(config="./sample_config_prior.yml")
priors = P.get_priors()
print(priors)

if __name__ == '__main__':
    P = PriorEngine(config="./sample_config_prior.yml")
    priors = P.get_priors()
    print(priors)
