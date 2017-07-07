"""
Prior Engine
"""

import tempfile
import os


class PriorEngine(object):

    def __init__(self, **kwargs):
        self.config = kwargs.get('config', None)
        self.priors = kwargs.get('priors', None)

    def _check(self):
        assert self.config is not None
        assert self.priors is not None

    def get_priors(self):
        res = {}
        for p in self.priors.keys():
            res.update({p: self._get_prior(p)})
        return res

    def _get_prior(self, p):
        if p == 'sm':
            prior = SoilMoisturePrior(ptype=self.priors[p]['type'])
        elif p == 'vegetation':
            assert False, \
                'The veg prior should provide cross-correlated prior information of vegetation characteristics'
        elif p == 'roughness':
            prior = RoughnessPrior(ptype=self.priors[p]['type'], lc_file=self.config.landcover,
                                   lut_file=self.config.luts['roughness'])
        else:
            assert False, 'Invalid prior'

        # calculate prior
        prior.calc()

        return prior.file  # return filename where prior file is located


class Prior(object):
    def __init__(self, **kwargs):
        self.ptype = kwargs.get('ptype', None)
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
            self.file = self._get_climatology_file()
        elif self.ptype == 'recent':
            self.file = self._get_recent_sm_proxy()
        else:
            assert False

    def _get_climatology_file(self):
        """
        return filename of preprocessed climatology
        """
        f = tempfile.mktemp()
        os.system('touch ' + f)
        return f

    def _get_recent_sm_proxy(self):
        assert False


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
