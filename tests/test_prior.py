# import sys
import os
import pytest

import tempfile
from multiply_prior_engine import PriorEngine, SoilMoisturePrior


def test_priorengine_init():
    P = PriorEngine(config='./tests/test_config_prior.yml')

    assert P.configfile is not None
    assert type(P.configfile) is str
    assert type(P.priors) is dict


def test_priorengine_get_priors():
    P = PriorEngine(config='./tests/test_config_prior.yml')
    assert type(P.get_priors()) is dict


def test_sm_prior_init():
    with pytest.raises(AssertionError,
                       message=("Expecting AssertionError \
                                --> no config specified")):
        SoilMoisturePrior()


def test_sm_prior_no_ptype():
    with pytest.raises(AssertionError,
                       message=("Expecting AssertionError \
                                --> no config specified")):
        SoilMoisturePrior()


def test_sm_prior_invalid_ptype():
    with pytest.raises(AssertionError,
                       message=("Expecting AssertionError \
                                --> no config specified")):
        SoilMoisturePrior(ptype='climatologi')


def test_calc_config():
    P = PriorEngine(config='./tests/test_config_prior.yml')
    S = SoilMoisturePrior(config=P.config,
                          ptype='climatology')
    assert type(S.config) is dict


def test_calc_output():
    P = PriorEngine(config='./tests/test_config_prior.yml')
    S = SoilMoisturePrior(config=P.config,
                          ptype='climatology')
    S.calc()
    assert os.path.exists(S.file)


# def test_roughness():
#     lut_file = tempfile.mktemp(suffix='.lut')
#     lc_file = tempfile.mktemp(suffix='.nc')
#     gen_file(lut_file)
#     gen_file(lc_file)
#     P = RoughnessPrior(ptype='climatology',
#                        lut_file=lut_file,
#                        lc_file=lc_file)
