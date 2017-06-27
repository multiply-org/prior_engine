import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + '..' )
from multiply_prior_engine import SoilMoisturePrior

class TestPrior(unittest.TestCase):
    def test_sm_prior(self):
        S = SoilMoisturePrior(ptype='climatology')

    def test_calc(self):
        S = SoilMoisturePrior(ptype='climatology')
        S.calc()
        print S.file
        self.assertTrue(os.path.exists(S.file))

