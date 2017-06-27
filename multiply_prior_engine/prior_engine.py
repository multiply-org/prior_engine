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
            res.update({p : self._get_prior(p)})
        return res

    def _get_prior(self, p):
        if p == 'sm':
            P = SoilMoisturePrior(ptype=self.priors[p]['type'])
        else:
            assert False, 'Invalid prior'

        # calculate prior
        P.calc()

        return P.file  # return filename where prior file is located




class Prior(object):
    def __init__(self, **kwargs):
        self.ptype = kwargs.get('ptype', None)
        self._check()

    def _check(self):
        assert self.ptype is not None, 'Invalid prior type'

    def calc(self):
        assert False, 'Should be implemented in child class'


class SoilMoisturePrior(Prior):
    def __init__(self, **kwargs):
        super(SoilMoisturePrior, self).__init__(**kwargs)

    def calc(self):
        if self.ptype == 'climatology':
            self.file = self._get_climatology_file()
        else:
            assert False

    def _get_climatology_file(self):

        f = tempfile.mktemp()
        os.system('touch ' + f)
        return f




