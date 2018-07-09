from .soilmoisture_prior_creator import SoilMoisturePriorCreator, RoughnessPriorCreator
from .vegetation_prior_creator import VegetationPriorCreator
from .prior_engine import PriorEngine
from .version import *
import logging
import tempfile
import os
from .prior_logger import PriorLogger

# Setup of temporary directory
# ----------------------------------
tempfile.tempdir = os.path.join(tempfile.gettempdir(),
                                'MULTIPLYPriorEngine')
try:
    os.mkdir(tempfile.tempdir)
except:
    pass

# Setup logging
# -----------------
# Default:
#   - PriorLogger() with default values:
#     - level='warning'
#     - handlers=['console', 'file']

PriorLogger(level='debug', handlers=['console'])

logging.info('The temporary directory is set to {}'.format(tempfile.tempdir))
