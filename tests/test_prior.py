import unittest
import sys
import os

import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + '..' )
from multiply_prior_engine import SoilMoisturePrior, RoughnessPrior

class TestPrior(unittest.TestCase):

    def gen_file(self, s):
        os.system('touch ' + s)
    
    def test_sm_prior(self):
        S = SoilMoisturePrior(ptype='climatology')

    def test_calc(self):
        S = SoilMoisturePrior(ptype='climatology')
        S.calc()
        print S.file
        self.assertTrue(os.path.exists(S.file))


    def test_roughness(self):
        lut_file = tempfile.mktemp(suffix='.lut')
        lc_file = tempfile.mktemp(suffix='.nc')
        self.gen_file(lut_file)
        self.gen_file(lc_file)
        P = RoughnessPrior(ptype='climatology', lut_file=lut_file, lc_file=lc_file)

        P.calc()



