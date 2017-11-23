#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Soil Moisture Prior Engine for MULTIPLY.

    Copyright (C) 2017  Thomas Ramsauer
"""


import datetime
import os
import tempfile

import numpy as np
import shapely
import shapely.wkt
import yaml
from netCDF4 import Dataset
from scipy import spatial


__author__ = ["Alexander Löw", "Thomas Ramsauer"]
__copyright__ = "Copyright 2017, Thomas Ramsauer"
__credits__ = ["Alexander Löw", "Thomas Ramsauer"]
__license__ = "GPLv3"
__version__ = "0.0.1"
__maintainer__ = "Thomas Ramsauer"
__email__ = "t.ramsauer@iggf.geo.uni-muenchen.de"
__status__ = "Prototype"


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
        """
        Get prior data.
        calls *_get_prior* for all priors in config.

        :returns: dictionary with prior names/(state vector,
                  inverse covariance matrix) as key/value
        :rtype: dictionary

        """
        res = {}
        # for p in self.priors.keys():
        for p in self.priors:
            res.update({p: self._get_prior(p)})

        return self._concat_priors(res)

    def _get_config(self):
        """
        Load confing from self.configfile.
        writes to self.config.

        :returns: nothing
        """
        with open(self.configfile, 'r') as cfg:
            self.config = yaml.load(cfg)
        assert self.config['Prior'] is not None, \
            ('There is no prior config information in {}'
             .format(self.configfile))

    def _get_prior(self, p):
        """ Called by get_priors for all prior keys in config.
        For specific prior (e.g. sm_clim) get prior info and calculate
        prior.

        :param p: prior name (e.g. sm_clim)
        :returns: prior file
        :rtype:

        """
        if p[:2] == 'sm':
            # assert self.priors[p]['type'] is not None, \
            ptype = self.config['Prior']['sm'][p]['type']
            assert ptype is not None, \
                'No prior type for soil moisture prior specified!'

            # pass config and prior type to subclass
            prior = SoilMoisturePrior(ptype=ptype,
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
        # state_vector, c_prior_inv = prior.calc()

        return prior.calc()

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
            temp_dict = {k: v for (k, v) in prior_dict.items() if key in k}
            res_concat.update({key: list(zip(temp_dict.values()))})

        return res_concat


class Prior(object):
    def __init__(self, **kwargs):
        self.ptype = kwargs.get('ptype', None)
        self.config = kwargs.get('config', None)
        self._check()

    def _check(self):
        assert self.ptype is not None, 'Invalid prior type'
        assert self.config is not None, 'No config available.'

    def calc(self):
        assert False, 'Should be implemented in child class'


class MapPrior(Prior):
    """
    Prior which is based on a LC map and a LUT
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
    Soil moisture prior class.
    Calculation of climatological prior.
    """
    def __init__(self, **kwargs):
        super(SoilMoisturePrior, self).__init__(**kwargs)

    def calc(self):
        """
        Initialize prior specific (climatological, ...) calculation.

        :returns: nothing
        """
        if self.ptype == 'climatology':
            # self.prior = self._calc_climatological_prior()
            return self._calc_climatological_prior()
        # elif self.ptype == 'recent':
            # self.file = self._get_recent_sm_proxy()
        else:
            assert False, '{} prior for sm not implemented'.format(self.ptype)

    def _get_climatology_file(self):
        """
        Load pre-processed climatology into self.clim_data.
        Part of prior._calc_climatological_prior().

        """
        assert (self.config['Prior']['sm']['sm_clim']
                           ['climatology_file']) is not None,\
            'There is no climatology file specified in the config!'

        # use xarray:
        # self.clim_data = xr.open_dataset(self.config['priors']['sm_clim']
        self.clim_data = Dataset(self.config['Prior']['sm']['sm_clim']
                                 ['climatology_file'])

    def _extract_climatology(self):
        """
        Extract climatology values for ROI.
        Part of _clac_climatological_prior().

        """
        clim = self.clim_data.variables['sm'][:]
        std = self.clim_data.variables['sm_stdev'][:]
        lats = self.clim_data.variables['lat'][:]
        lons = self.clim_data.variables['lon'][:]

        ROI_wkt = shapely.wkt.loads(self.config['General']['roi'])

        # minx, miny, maxx, maxy:
        minx, miny, maxx, maxy = ROI_wkt.bounds
        ROI = [(minx, miny), (maxx, maxy)]

        # TODO insert check for ROI bounds:
        # if POI[0] < np.amin(lats) or POI[0] > np.amax(lats) or\
        #    POI[1] < np.amin(lons) or POI[1] > np.amax(lons):
        #     raise ValueError("POI's latitude and longitude out of bounds.")

        # stack all raveled lats and lons from climatology
        # combined_LAT_LON results in e.g.
        # [[ 34.625 -10.875]
        #  [ 34.625 -10.625]
        #  [ 34.625 -10.375]...]
        combined_LAT_LON = np.dstack([lats.ravel(), lons.ravel()])[0]
        mytree = spatial.cKDTree(combined_LAT_LON)
        idx, idy = [], []
        # for all coordinate pairs in ROI search indexes in combined_LAT_LON:
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

        # TODO Respect spatial resolution in config file.
        #      Adjust result accordingly.

        # print(sm_area)
        self.clim = sm_area
        self.std = sm_area_std

    def _calc_climatological_prior(self):
        """
        Calculate climatological prior.
        Reads climatological file and extracts proper values for given
        timespan and -interval.
        Then converts the means and stds to state vector and covariance
        matrices.

        :returns: state vector and covariance matrix
        :rtype: tuple

        """
        self._get_climatology_file()
        self._extract_climatology()

        # TODO limit to months

        # date_format = ('%Y-%m-%d')
        s = self.config['General']['start_time']
        e = self.config['General']['end_time']
        t_span = (e-s).days + 1
        # print(t_span)

        # create list of month ids for every queried point in time:
        idt = [(s+datetime.timedelta(x)).month
               for x in range(t_span)]
        # idt_unique = list(set(idt))

        # create nd array with correct dimensions (time, x, y):
        p = np.ndarray(shape=(len(idt), self.clim.shape[1],
                              self.clim.shape[2]),
                       dtype=float)
        std = p.copy()

        # read correspending data into mean and std arrays:
        for i in range(len(idt)):
            p[i, :, :] = self.clim[idt[i]-1, :, :]
            std[i, :, :] = self.std[idt[i]-1, :, :]

        # calculate uncertainty with normalization via coefficient of variation
        # TODO scale uncertainty
        sm_unc = (std/np.mean(self.clim))
        # inverse covariance matrix
        diagon = (1./sm_unc)
        # print(diagon.shape)

        # def create_sparse_matrix(a):
        #     return sp.sparse.lil_matrix(np.eye(t_span)*a)

        # C_prior_inv = np.apply_along_axis(create_sparse_matrix, 0, diagon)
        C_prior_inv = diagon

        # DISCUSS TODO
        # rather write to self.'prior_key' to easy concatenate afterwards
        # via concat_priors.

        return p, C_prior_inv

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


def get_mean_state_vector(config_file):
    """
    Return state vector and inverse covariance matrix for priors.

    :param config_file: path to config file
    :returns: state vector and inverse covariance matrix
    :rtype: dictionary

    """
    return PriorEngine(config="./sample_config_prior.yml").get_priors()


if __name__ == '__main__':
    get_mean_state_vector(config_file="./sample_config_prior.yml")
