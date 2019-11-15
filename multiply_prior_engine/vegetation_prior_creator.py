# /usr/bin/env python
__author__ = "J Timmermans"
__copyright__ = "Copyright 2017 J Timmermans"
__email__ = "j.timmermans@cml.leidenuniv.nl"

import glob
import logging
import os
import time

import multiprocessing
import numpy as np
import yaml
from dateutil.parser import parse
from matplotlib import pyplot as plt
from multiply_core.util import get_aux_data_provider
from scipy import interpolate as RegularGridInterpolator
from netCDF4 import Dataset
from shapely.wkt import loads
import gdal
from .prior_creator import PriorCreator

SUPPORTED_VARIABLES = ['lai', 'cab', 'cb', 'car', 'cw', 'cdm', 'n',
                       'ala', 'h', 'bsoil', 'psoil']


plt.ion()


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    """
    Enable Parallel processing
    This code is created to enable parallel processing with python

    :param f: function to be called
    :param X: input to the function
    :param nprocs: number of cores to be used
    :returns: output of function
    :rtype:

    """
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


def _get_config(configfile):
    """
    Load config from self.configfile.
    writes to self.config.

    :returns: -
    """
    try:
        with open(configfile, 'r') as cfg:
            config = yaml.load(cfg)
    except FileNotFoundError as e:
        logging.info('Info: current directory: {}'.format(os.getcwd()))
        logging.error('{}'.format(e.args[0]))
        raise
    try:
        assert 'Prior' in config.keys(),\
            ('There is no prior section in configuration file ({}).'
             .format(configfile))
        assert config['Prior'] is not None,\
            ('There is no prior configuration in the config file ({}).'
             .format(configfile))
    except AssertionError as e:
        logging.error('{}'.format(e.args[0]))
        raise
    return config


def processespercore(varname, PFT, PFT_ids, VegetationPriorCreator):
    """
    Create Prior values from PFT distributions and Vegetation traits
    For each PFT the specific trait (according to varname) are read from the Trait-Database. These traits are
    then statistically analysed to produce the mean and standard deviations. These trait values are then evaluated
    against the PFT distribution (occurrence) map and joint together to create a single Prior (mean&uncertainty)
    estimate for each spatial location

    Please note that: This function is encapsulated within the parmap method to run in parallel on different cores

    :param varname: variable to be processed
    :param PFT: arrays containing global Maps of PFT distributions
    :param PFT_ids: a list containing PFT ids
    :param VegetationPriorCreator: class containing all the functionality to be run (per core)
    :returns: Vegetation Prior average values, Vegetation Prior uncertainty values
    :rtype:

    """
    TRAIT_ttf_avg = PFT[:, :, 0].astype('float') * 0.
    TRAIT_ttf_unc = PFT[:, :, 0].astype('float') * 0.

    for pft_id in PFT_ids[1:]:
        PFT_id = PFT[:, :, pft_id]

        # extract statistical values for transformed variables
        at = time.time()
        b = time.time()

        if np.any(PFT_id):
            trait_id = VegetationPriorCreator.ReadTraitDatabase(varname, pft_id)

            # filtering variables for erroneous values
            trait_f_ = trait_id[varname] * 1.

            # transform variables
            trait_tf_ = trait_f_
            try:
                trait_tf_ = VegetationPriorCreator.transformations[varname](trait_f_)
            except:
                trait_tf_ = trait_f_


            ierror = ~np.isnan(trait_tf_)
            trait_tf_avg = np.mean(trait_tf_[ierror])
            trait_tf_unc = np.std(trait_tf_[ierror])
            trait_tf_unc = np.max([trait_tf_unc,1e-9])

            # assign PFT weights of individual pft_id to avg/unc values
            TRAIT_wtf_avg = PFT_id / 100. * trait_tf_avg
            TRAIT_wtf_unc = PFT_id / 100. * trait_tf_unc

            # adding two (large-matrixes) is very computationally intensive
            TRAIT_ttf_avg = TRAIT_ttf_avg + TRAIT_wtf_avg
            TRAIT_ttf_unc = TRAIT_ttf_unc + TRAIT_wtf_unc

            # print(varname)
            # print([PFT_id[590,2864], PFT_id[2864,590]])
            # print(np.sum(~ np.isnan(trait_f_)))
            # print(trait_tf_avg)
            # print(trait_tf_unc)

    # post processes for erroneous data
    corr_weight = np.sum(PFT[:,:,1:] / 100., axis=2)
    ierror = (TRAIT_ttf_avg == 0) + (TRAIT_ttf_unc == 0) + (corr_weight ==0)
    TRAIT_ttf_avg[ierror] = np.NaN
    TRAIT_ttf_unc[ierror] = np.NaN

    # print(np.nanmean(TRAIT_ttf_avg))
    # In some cases the sum of all PFT <>1. Therefore we are required to correct for this
    # causes for this are
    # 1) wetlands, where a PFT contains both water and if everything went correctly. Within Try, there are traits provided for water pfs.
    # 2) rounding errors. In the creation of PFT abundance
    TRAIT_ttf_avg = TRAIT_ttf_avg/ (corr_weight+ 1e-10)
    TRAIT_ttf_unc = TRAIT_ttf_unc / (corr_weight+ 1e-10)

    # print(varname)
    # print(np.nanmean(TRAIT_ttf_avg))
    # print(np.nanmean(TRAIT_ttf_unc))

    return TRAIT_ttf_avg, TRAIT_ttf_unc


class VegetationPriorCreator(PriorCreator):
    """
    Description
    """
    def __init__(self, **kwargs):
        super(VegetationPriorCreator, self).__init__(**kwargs)
        self.config = kwargs.get('config', None)
        self.priors = kwargs.get('priors', None)

        # 1. Parameters
        # 1.1 Define Study Area
        self.tileres = 10
        try:
            roi = loads(self.config['General']['roi'])
            min_lon = np.floor(roi.bounds[0] / self.tileres) * self.tileres
            max_lon = np.ceil(roi.bounds[2] / self.tileres) * self.tileres
            min_lat = np.floor(roi.bounds[1] / self.tileres) * self.tileres
            max_lat = np.ceil(roi.bounds[3] / self.tileres) * self.tileres
            self.lat_study = [min_lat, max_lat]
            self.lon_study = [min_lon, max_lon]
        except:
            self.lon_study = [-180, 180]
            self.lat_study = [-90, 90]

        # 1.2 Define paths
        aux_data_provider = get_aux_data_provider()
        self.directory_data = self.config['Prior']['General']['directory_data']
        self.path2LCC_file = (self.directory_data + 'LCC/' + 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7_updated.nc')
        self.path2Climate_file = (self.directory_data + 'Climate/' + 'sdat_10012_1_20171030_081458445.tif')
        self.path2Meteo_file = (self.directory_data + 'Meteorological/' + 'Meteo_.nc')
        self.path2Trait_file = (self.directory_data + 'Trait_Database/' + 'Traits.nc')
        self.path2Traitmap_file = self.directory_data + 'Priors/' + 'Priors.nc'
        if not aux_data_provider.assure_element_provided(self.path2LCC_file):
            logging.warning(f'{self.path2LCC_file} could not be found!')
        if not aux_data_provider.assure_element_provided(self.path2Climate_file):
            logging.warning(f'{self.path2Climate_file} could not be found!')
        if not aux_data_provider.assure_element_provided(self.path2Trait_file):
            logging.warning(f'{self.path2Trait_file} could not be found!')

        self.output_directory = self.config['Prior']['output_directory']
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)

        self.plotoption = 0  # [0,1,2,3,4,5,6,..]

        # define Prior values that are not found in online databases.
        self.mu = dict()
        self.unc = dict()
        self.mu['cbrown'] = 0.2
        self.mu['bsoil'] = 0.5
        self.mu['psoil'] = 0.5

        # add user defined priors (mu and/or unc)
        for varname in SUPPORTED_VARIABLES:
            if varname in self.config['Prior']:
                if 'user' in self.config['Prior'][varname]:
                    user_def = self.config['Prior'][varname]['user']
                    if 'mu' in user_def:
                        self.mu[varname] = user_def['mu']

                    if 'unc' in user_def:
                        self.unc[varname] = user_def['unc']

    # 0. Define parameter transformations
        self.transformations = dict({
            'lai': lambda x: np.exp(-x / 2.),
            'cab': lambda x: np.exp(-x / 100.),
            'car': lambda x: np.exp(-x / 100.),
            'cw': lambda x: np.exp(-50. * x),
            'cdm': lambda x: np.exp(-100. * x),
            'ala': lambda x: x / 90.})
        self.inv_transformations = dict({
            'lai': lambda x: -2. * np.log(x),
            'cab': lambda x: -100 * np.log(x),
            'car': lambda x: -100 * np.log(x),
            'cw': lambda x: (-1 / 50.) * np.log(x),
            'cdm': lambda x: (-1 / 100.) * np.log(x),
            'ala': lambda x: 90. * x})


    @classmethod
    def get_variable_names(cls):
        return SUPPORTED_VARIABLES

    # General structure of processing chains
    def ProcessData(self, variables=None, state_mask=None,
                    timestr='2005-05-05 05:55', logger=None, file_prior=None,
                    file_lcc=None, file_biome=None, file_meteo=None):
        """
        Process Data
        Apriori Calculation of prior using Databases of Vegetation Traits.
        This function is split into two parts (which are run for all Tiles over the study are)
        - OfflineProcessing: This only has to be performed once (to make sure all the input data is available)
        - StaticProcessing:  Creating Peak Biomass (PBM) Priors
        - DynamicProcessing: Extending PBM traits to seasonal priors (a placeholder for the later implementations)

        :param variables: list of variables to be converted into global file
        :param state_mask: place holder for spatial mask (not implemented)
        :param timestr: string containing date&time '2007-12-31 04:23' for which global file needs to be created
        :param logger: log-file for capturing message from the scripts
        :param file_prior: place-holder for prior (TRY) database - filename (at the moment hardcoded)
        :param file_lcc: place-holder for landcover data - filename (at the moment hardcoded)
        :param file_biome: place-holder for biome data - filename (at the moment hardcoded)
        :param file_meteo: place-holder for meteorological data - filename (at the moment hardcoded)

        :returns: filenames to global VRT prior files
        :rtype:

        """
        if type(variables) is str:
            variables = [variables]

        # offline processing (to be run only once)
        # self.OfflineProcessing()


        timea = datetime.datetime.now()
        plt.ion()

        # # Define variables
        # if variables==None:
        #     variables = ['lai', 'cab', 'cb', 'car', 'cw', 'cdm', 'N', 'ala',
        #                  'h', 'bsoil', 'psoil']
        #
        # # 1.1 Define paths
        # directory_data = '/home/joris/Data/Prior_Engine/'
        # if file_prior is None:
        #     file_prior = directory_data + 'Trait_Database/' + 'Traits.nc'
        # if file_lcc is None:
        #     file_lcc = (directory_data +'/LCC/'
        #                 + 'ESACCI-LC-L4-LCCS-Map-300m-P1Y'
        #                 + '-2015-v2.0.7_updated.nc')
        # if file_biome==None:
        #     file_biome = (directory_data +'Climate/'
        #                   + 'sdat_10012_1_20171030_081458445.tif')
        # if file_meteo==None:
        #     file_meteo = directory_data +'Meteorological/' + 'Meteo_.nc'
        # file_output = directory_data +'Priors/' + 'Priors.nc'
        #
        # 0. Setup Processing
        # VegPrior = VegetationPrior()

        # VegPrior.path2Trait_file = file_prior
        # VegPrior.path2LCC_file = file_lcc
        # VegPrior.path2Climate_file = file_biome
        # VegPrior.path2Meteo_file = file_meteo

        #############
        time = parse(timestr)
        doystr = time.strftime('%j')

        minlon  =  np.min(self.lon_study)
        maxlon  =  np.max(self.lon_study)
        minlat =   np.min(self.lat_study)
        maxlat =   np.max(self.lat_study)

        lat_study_ = np.arange(np.floor(minlat / self.tileres) , np.ceil(maxlat / self.tileres) ) * self.tileres
        lon_study_ = np.arange(np.floor(minlon / self.tileres) , np.ceil(maxlon / self.tileres) ) * self.tileres

        for lon_study in lon_study_:
            for lat_study in lat_study_:
                print('%3.2f %3.2f' % (lon_study, lat_study))

                self.lon_study = [lon_study, lon_study + 10]
                self.lat_study = [lat_study, lat_study + 10]

                # 3. Perform Static processing
                lon, lat, Prior_pbm_avg, Prior_pbm_unc = \
                  self.StaticProcessing(variables)

                # 4. Perform Static processing
                self.DynamicProcessing(variables, lon, lat, Prior_pbm_avg,
                                                   Prior_pbm_unc, doystr=doystr)

        # filenames = self.CombineTiles2Virtualfile(variables, doystr)

    def OfflineProcessing(self):
        """

        Creation of LCC landcover map
        This
        :returns:
        :rtype:

        """
        # handle offline

        # Download Data
        # self.DownloadCrossWalkingTable()

        # Preprocess Data
        # self.RunCrossWalkingTable()

        # Construct Database
        # self.CreateDummyDatabase()
        self.CreateRealDatabase()

    def StaticProcessing(self, varnames, write_output=False):
        """
        Creating Peak Biomass (PBM) Priors
        Priors are created by upscaling vegetation traits obtained through the TRY database. Within the TRY database
        vegetation traits are provided per PFT group. In order to upscale these values, a global PFT map is required.
        This is created by merging a global Landcover map (from Climate Change Initiative, CCI) with a climate zone
        map (using the Koppen classification). This is accomplished by
        -ReadLCC: Reading the CCI Landcover map
        -ReadClimate: Reading the Koppen Climate zone map
        -RescaleCLM: Rescaling Climate zone map to collocate with Landcover CCI.
        -Combine2PFT: Combining Climate zone + Landcover maps into PFTs
        Using this global PFT map, the values from the TRY database are afterwards spatially distributed by
        -AssignPFTTraits2Map: assigning and aggregating traits to PFT maps.

        :param varnames: list of variables to be converted into global file
        :param write_output: Binary value (TRUE/FALSE) controlling the writing of outputfiles
        :returns: longitude, latitude, Prior_avg, Prior_unc
        :rtype:

        """
        # Read Data (2.5s)
        LCC_map, LCC_lon, LCC_lat, LCC_classes = self.ReadLCC()
        CLM_map, CLM_lon, CLM_lat, CLM_classes = self.ReadClimate()

        if np.all((LCC_map['Water'] + LCC_map['Snow_Ice'])>80):

            Prior_pbm_avg = dict()
            Prior_pbm_unc = dict()
            dummy = LCC_map['Water'] * np.NaN

            for varname in varnames:
                Prior_pbm_avg[varname] = dummy
                Prior_pbm_unc[varname] = dummy

        else:
            # Process Data (12.5s)
            CLM_map_i = self.RescaleCLM(CLM_lon, CLM_lat, CLM_map, LCC_lon, LCC_lat)
            PFT, PFT_classes, Npft, PFT_ids = self.Combine2PFT(LCC_map,  CLM_map_i)

            Prior_pbm_avg, Prior_pbm_unc = self.AssignPFTTraits2Map(PFT,PFT_ids,varnames)

        if write_output:
            # self.WriteOutput(LCC_lon, LCC_lat, Prior_pbm_avg, Prior_pbm_unc)
            self.WriteGeoTiff(LCC_lon, LCC_lat, Prior_pbm_avg, Prior_pbm, unc)

        return LCC_lon, LCC_lat, Prior_pbm_avg, Prior_pbm_unc

    def DynamicProcessing(self, varnames, LCC_lon, LCC_lat, Prior_pbm_avg,
                          Prior_pbm_unc, doystr, write_output=True):
        """
        Extending Peak Biomass (PBM) traits to seasonal Priors
        At this moment, this function is only a placeholder for the later
        implementations. The final implementation will be modelled using
        - covariances between traits and (seasonal) meteorological variables
        - phenological evolution (trained using plant growth models)

        :param varnames: list of variables to be converted into global file
        :param LCC_lon:  array with longitude values of (subsetted tile of) study area
        :param LCC_lat: array with latitude values of (subsetted tile of) study area
        :param Prior_pbm_avg: Vegetation Traits mean value at PBM
        :param doystr: string containing date&time '2007-12-31 04:23' for processing needs to be performed
        :param Prior_pbm_unc: Vegetation Traits uncertainty value at PBM
        :param write_output: Binary Value (TRUE/FALSE) controlling the writing of outputfiles
        :returns: -
        :rtype:

        """

        Meteo_map, Meteo_lon, Meteo_lat = self.ReadMeteorologicalData(doystr)
        # Meteo_map_i =  RescaleCLM(Meteo_lon, Meteo_lat, Meteo_map,
        #                           LCC_lon, LCC_lat)

        Prior_avg, Prior_unc = self.PhenologicalEvolution(Prior_pbm_avg,
                                                          Prior_pbm_unc,
                                                          doystr,
                                                          Meteo_map_i=None)

        # 7. Write Output
        if write_output:
            self.WriteGeoTiff(LCC_lon, LCC_lat, Prior_avg, Prior_unc, doystr)
            # self.WriteOutput(LCC_lon, LCC_lat, Prior_avg, Prior_unc, doystr)


    # Reading data functions
    def ReadLCC(self):
        """
        Read Landcover information
        The Landcover map from the Climate Change Initiaive (CCI) is read.

        :returns: landcover map, longitude, latitude, landcover class names
        :rtype:

        """
        lon_min = self.lon_study[0]
        lon_max = self.lon_study[1]
        lat_min = self.lat_study[0]
        lat_max = self.lat_study[1]

        dataset_container = Dataset(self.path2LCC_file, 'r')
        lon = dataset_container.variables['lon'][:]
        lat = dataset_container.variables['lat'][:]

        ilon_min = np.argmin(np.abs(lon - lon_min))
        ilon_max = np.argmin(np.abs(lon - lon_max))
        ilat_min = np.argmin(np.abs(lat - lat_min))
        ilat_max = np.argmin(np.abs(lat - lat_max))

        lon_s = lon[ilon_min:ilon_max]
        lat_s = lat[ilat_max:ilat_min]

        # classes0 = dataset_container['lccs_class'][ilon_min,ilat_min]
        class_names = ['Tree_Broadleaf_Evergreen',
                       'Tree_Broadleaf_Deciduous',
                       'Tree_Needleleaf_Evergreen',
                       'Tree_Needleleaf_Deciduous',
                       'Shrub_Broadleaf_Evergreen',
                       'Shrub_Broadleaf_Deciduous',
                       'Shrub_Needleleaf_Evergreen',
                       'Shrub_Needleleaf_Deciduous',
                       'Natural_Grass', 'Managed_Grass',
                       'Bare_Soil', 'Water', 'Snow_Ice']

        Data = dict()
        for class_name in class_names:
            data = dataset_container[class_name][ilat_max:ilat_min,
                                                 ilon_min:ilon_max]
            Data[class_name] = data

        if self.plotoption == 1:
            Nc = 4
            Nv = len(Data)
            Nr = int(np.ceil(Nv / Nc + 1))

            plt.figure(figsize=[20, 20])
            for i, class_name in enumerate(class_names):
                plt.subplot(Nr, Nc, i + 1)
                plt.imshow(Data[class_name])
                plt.title(class_name)
                plt.colorbar()

        return Data, lon_s, lat_s, class_names

    def ReadClimate(self):
        """
        Read Climate Zone information
        A Climate Zone map (created on basis of the Koppen Climatic Zone classification)is read.

        :returns: climate zone map, longitude, latitude, climate zone classes
        :rtype:

        """

        ds = gdal.Open(self.path2Climate_file)
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()

        minx = gt[0]
        miny = gt[3] + width * gt[4] + height * gt[5]
        maxx = gt[0] + width * gt[1] + height * gt[2]
        maxy = gt[3]

        minx = np.round(minx * 100) / 100
        miny = np.round(miny * 100) / 100
        maxx = np.round(maxx * 100) / 100
        maxy = np.round(maxy * 100) / 100

        lon = np.linspace(minx, maxx, width + 1)[0:-1]
        lat = np.linspace(maxy, miny, height + 1)[0:-1]

        lon_min = self.lon_study[0]
        lon_max = self.lon_study[1]
        lat_min = self.lat_study[0]
        lat_max = self.lat_study[1]

        ilon = np.where((lon >= lon_min) * (lon <= lon_max))[0]
        ilat = np.where((lat >= lat_min) * (lat <= lat_max))[0]

        # read data
        #data = ds.ReadAsArray(ilon[0], ilat[0], len(ilon), len(ilat))
        data = ds.ReadAsArray()[ilat,:][:,ilon]#(ilon[0], ilat[0], len(ilon), len(ilat))

        lon_s = lon[ilon]
        lat_s = lat[ilat]

        classes = ['H20',  # Water                                        0
                   'Af',  # Tropical/rainforest                           1
                   'Am',  # Tropical Monsoon                              2
                   'Aw',  # Tropical/Savannah                             3
                   'BWh',  # Arid/Desert/Hot                              4
                   'BWk',  # Arid/Desert/Cold                             5
                   'BSh',  # Arid/Steppe/Hot                              6
                   'BSk',  # Arid/Steppe/Cold                             7
                   'Csa',  # Temperate/Dry_Symmer/Hot_summer              8
                   'Csb',  # Temperate/Dry_Symmer/Warm_summer             9
                   'Csc',  # Temperate/Dry_Symmer/Cold_summer             10
                   'Cwa',  # Temperate/Dry_Winter/Hot_summer              11
                   'Cwb',  # Temperate/Dry_Winter/warm_summer             12
                   'Cwc',  # Temperate/Dry_Winter/Cold_summer             13
                   'Cfa',  # Temperate/Without_dry_season/Hot_summer      14
                   'Cfb',  # Temperate/Without_dry_season/Warm_summer     15
                   'Cfc',  # Temperate/Without_dry_season/Cold_summer     16
                   'Dsa',  # Cold/Dry_Summer/Hot_summer                   17
                   'Dsb',  # Cold/Dry_Summer/warm_summer                  18
                   'Dsc',  # Cold/Dry_Summer/cold_summer                  19
                   'Dsd',  # Cold/Dry_Summer/very_cold_summer             20
                   'Dwa',  # Cold/Dry_Winter/Hot_summer                   21
                   'Dwb',  # Cold/Dry_Winter/Warm_summer                  22
                   'Dwc',  # Cold/Dry_Winter/Cold_summer                  23
                   'Dwd',  # Cold/Dry_Winter/Very_cold_summer             24
                   'Dfa',  # Cold/Without_dry_season/Hot_summer           25
                   'Dfb',  # Cold/Without_dry_season/Warm_summer          26
                   'Dfc',  # Cold/Without_dry_season/cold_summer          27
                   'Dfd',  # Cold/Without_dry_season/very_cold_summer     28
                   'ET(1)',  # Polar/Tundra                               29
                   'EF(1)',  # Polar/Frost                                30
                   'ET(2)',  # Polar/Tundra                               31
                   'EF(2)'  # Polar/Frost                                 32
                   ]

        if self.plotoption == 3:
            plt.figure(figsize=[20, 20])
            plt.imshow(data)
            plt.title('Koppen')

        return data, lon_s, lat_s, classes

    def ReadTraitDatabase(self, varnames, pft_id=1):
        """
        Read Traits from Database
        A local (modified) version of the Try Database (containing vegetation traits) is read.

        :param varnames: list of variables to be converted into global file
        :param pft_id: list of pft id numbers for which the traits needs to be read.
        :returns: an array of Traits per PFT group
        :returns: an array of Traits per PFT group
        :rtype:

        """
        if type(varnames ) is str:
            varnames = [varnames]

        # Read which variables are within the database file
        Data = Dataset(self.path2Trait_file, 'r')
        vars_in_database = Data.variables.keys()
        Data.close()

        Var = dict()
        for varname in varnames:

            # create normal distribution around mu[varname]
            if (varname in self.mu):
                if (varname in self.unc):
                    std = self.unc[varname]
                else:
                    std = self.mu[varname]*0.001
                V = np.random.normal(loc=self.mu[varname], scale=std, size=100)
                # ensure that none of the vegetation priors have (real) values <0
                V * (V > 1e-9) + 1e-9 * (V < 1e-9)

            # use try database values
            elif varname in vars_in_database:
                Data = Dataset(self.path2Trait_file, 'r')
                V = Data[varname][:, pft_id, :]

                # if too few entries were available for TRY for a specific PFT, we define the Prior for this PFT to be equivalent to average of all values of this variable
                if np.sum(~ np.isnan(V)) < 10:
                    V = Data[varname][:, :, :]
                Data.close()

                # use mean value from database, but with user-specified uncertainty
                if (varname in self.unc):
                    mu = np.nanmean(V)
                    std = self.unc[varname]
                    V = np.random.normal(loc=mu, scale=std, size=100)

                    # ensure that none of the vegetation priors have (real) values <0
                    V * (V > 1e-9) + 1e-9 * (V < 1e-9)

            # parameter is not embedded in database or specified by user
            else:
                V = np.array([1., 1. - 1e-5])

            Var[varname] = V

        return Var

    def ExtractPFT4TryDatabaseEntries(self, Lat_, Lon_, Plantgroup_, Crop_, LeafType_, C3C4_, LeafPhen_):
        Lat = np.floor(Lat_.astype(float)*1000)/1000.
        Lon = np.floor(Lon_.astype(float) * 1000) / 1000.

        LatLon = [complex(a, b) for a, b in zip(Lat, Lon)]
        LatLon_u = np.unique(LatLon)

        ID = np.zeros_like(Lat)+1
        for i,latlon_u in enumerate(LatLon_u):
            lon_u =  np.imag(latlon_u)
            lat_u = np.real(latlon_u)

            # tic = time.time()
            # ID = []
            # ID = np.zeros_like(Lat_)
            # this runs over the full length of TRY entries.. therefore
            # for i, lat in enumerate(Lat_):  # number of vaes is 31794

            # self.lat_study = [Lat_[i].astype('float') - 0.1, Lat_[i].astype('float') + 0.1]
            # self.lon_study = [Lon_[i].astype('float') - 0.1, Lon_[i].astype('float') + 0.1]

            self.lat_study = [lat_u - 0.1, lat_u + 0.1]
            self.lon_study = [lon_u - 0.1, lon_u + 0.1]

            # Obtain climate map
            CLM_map, CLM_lon, CLM_lat, CLM_classes = self.ReadClimate()

            # iwater                                      =   (CLM_map_i[36,36] ==  0)
            itropical = (CLM_map[0, 0] >= 1) * (CLM_map[0, 0] <= 7)
            itemporate = (CLM_map[0, 0] >= 8) * (CLM_map[0, 0] <= 16)
            iboreal = (CLM_map[0, 0] >= 17) * (CLM_map[0, 0] <= 28)
            ipolar = (CLM_map[0, 0] >= 29) * (CLM_map[0, 0] <= 32)

            # pft     =   PFT[36,36]      # pft from map

            id = 17
            if Crop_[i] == 'crop':
                if (C3C4_[i] == 'C3') + (C3C4_[i] == 'C3/CAM') + (C3C4_[i] == 'CAM') + (C3C4_[i] == 'C3/C4'):
                    id = 12
                elif (C3C4_[i] == 'C4') + (C3C4_[i] == 'C4/CAM'):
                    id = 13

            if Plantgroup_[i] == 'graminoid':
                if (C3C4_[i] == 'C3') + (C3C4_[i] == 'C3/CAM') + (C3C4_[i] == 'CAM') + (C3C4_[i] == 'C3/C4'):
                    if ipolar:  # arctic
                        id = 10
                    else:
                        id = 9
                elif (C3C4_[i] == 'C4') + (C3C4_[i] == 'C4/CAM'):
                    id = 11

            if (Plantgroup_[i] == 'tree') + (Plantgroup_[i] == 'shrub/tree'):
                if itropical:  # tropical
                    if (LeafPhen_[i] == 'deciduous') + (LeafPhen_[i] == 'deciduous/evergreen'):
                        id = 2
                    elif LeafPhen_[i] == 'evergreen':
                        id = 1
                elif itemporate:  # temperate
                    if (LeafType_[i] == 'needleleaved') + (LeafType_[i] == 'scale-shaped'):
                        id = 3
                    elif LeafType_[i] == 'broadleaved':
                        if (LeafPhen_[i] == 'deciduous') + (LeafPhen_[i] == 'deciduous/evergreen'):
                            id = 5
                        elif LeafPhen_[i] == 'evergreen':
                            id = 4
                elif iboreal:  # boreal
                    if (LeafPhen_[i] == 'deciduous') + (LeafPhen_[i] == 'deciduous/evergreen'):
                        if (LeafType_[i] == 'needleleaved') + (LeafType_[i] == 'scale-shaped'):
                            id = 8
                        elif LeafType_[i] == 'broadleaved':
                            id = 7
                    elif LeafPhen_[i] == 'evergreen':
                        id = 6

            if (Plantgroup_[i] == 'shrub') + (Plantgroup_[i] == 'fern') + (Plantgroup_[i] == 'herb') + (
                Plantgroup_[i] == 'herb/shrub'):  # shrubs
                id = 14

            if Plantgroup_[i] == 'moss':
                id = 15

            # ID.append(id)
            # ID[i] = id
            # I = np.where((lon_u == Lon) * (lat_u == Lat))
            I = np.where((lat_u == Lat)*(lon_u==Lon))
            ID[I] = id

        # toc = time.time()
        # print
        # toc - tic
        return ID

    def ReadTryDatabase(self):
        filename = 'EPTSv4_UTF8.csv'
        # database_path = 'C:/Users/Amie Corbin/Desktop/Data/Prior_Engine/Try_Database/' + filename
        import csv

        Rows = []
        aux_data_provider = get_aux_data_provider()
        path_to_file = os.path.abspath(os.path.join(self.directory_data, 'Try_Database', filename))
        if not aux_data_provider.assure_element_provided(path_to_file):
            logging.warning(f'Could not find {path_to_file}')
        with open(path_to_file, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            # spamreader = csv.reader(csvfile)
            # reader = csv.DictReader(csvfile)
            for row in spamreader:
                # print ', '.join(row)
                # row.replace("NA", "NaN")
                Rows.append(row)
        Rows = np.array(Rows)

        #site descriptions
        site = dict()
        site['Latitude'] = np.array([float(x.replace('NA', 'nan')) for x in Rows[1:,10]])
        site['Longitude'] = np.array([float(x.replace('NA', 'nan')) for x in Rows[1:,11]])
        site['PFT_id'] = np.array([int(x.replace('NA', 'nan')) for x in  Rows[1:, 8]])
        site['TRYSpeciesID'] = np.array([int(x.replace('NA', 'nan')) for x in  Rows[1:, 5]]) # TRYSpec_nu
        site['TRYSpeciesName']= Rows[1:,6] # Spec
        site['Plantgroup'] = Rows[1:,12] # Plantgroup
        site['Crop']    =  Rows[1:,13]  # 'Crop'
        site['LeafType'] =   Rows[1:,14]  # 'LeafType'
        site['C3C4'] = Rows[1:,15]  # 'Photosynth'
        site['LeafPhen'] = Rows[1:,16]  # 'LeafPhenol'
        site['Koppen'] = np.array([float(x.replace('NA', 'nan')) for x in Rows[1:,17]])  # 'Koppen Zone'

        # define order of traits on basis of the header
        TRYtraits_name = ['Cellulose', 'LAR', 'LWR', 'logLMA', 'Carbon', 'Phenols', 'Nitrogen', 'Phosphorus', \
                       'PlantHeight', 'LA', 'ala', 'cab', 'LDMC', 'logStarch', 'Lignin', 'car', 'cw', \
                       'Hydrogen', 'LeafThickness', 'Oxygen', 'N', 'LAI']
        TRYtraits_col = np.arange(18,40)

        # sort columns into their respective entries in a  dictionary
        traits = dict()
        for i, name in enumerate(TRYtraits_name):
            icol = TRYtraits_col[i]
            traits[name] = np.array([float(x.replace('NA', 'nan')) for x in Rows[1:, icol]])

        # post process logLMA to cdm (dry matter content) consistent with PROSAIL
        traits['cdm'] =np.exp(traits['logLMA'])

        id_traits = dict()
        id_traits['cdm'] = '11'
        id_traits['ala'] = '3'
        id_traits['cab'] = '413'
        id_traits['car'] = '491'
        id_traits['cw'] = '999001'
        id_traits['N'] = '999002'
        PFT_id = np.arange(1, 18)

        # sort the values into structure (used by the old version of the 'ReadTryFile' method)
        Latitude_, Longitude_, Koppen_,PFT_id_ = [np.array([]) for _ in range(4)]
        TRYSpeciesID_, TRYSpeciesName_, Plantgroup_ = [np.array([]) for _ in range(3)]
        Crop_, LeafType_, C3C4_, LeafPhen_ = [np.array([]) for _ in range(4)]
        TRYTraitName_, TRYTraitID_, TRYTrait_value_ = [np.array([]) for _ in range(3)]
        ierror_coor = (site['Latitude']<-90)+(site['Latitude']>90)+(site['Longitude']<-180)+(site['Longitude']>180)
        for trait_name in id_traits.keys():
            ierror_traits = np.isnan(traits[trait_name])
            ierror = ierror_traits + ierror_coor
            Latitude_ = np.hstack((Latitude_,site['Latitude'][~ierror]))
            Longitude_ = np.hstack((Longitude_, site['Longitude'][~ierror]))
            PFT_id_ = np.hstack((PFT_id_, site['PFT_id'][~ierror]))
            TRYSpeciesID_ = np.hstack((TRYSpeciesID_, site['TRYSpeciesID'][~ierror]))
            TRYSpeciesName_ = np.hstack((TRYSpeciesName_, site['TRYSpeciesName'][~ierror]))
            Plantgroup_ = np.hstack((Plantgroup_, site['Plantgroup'][~ierror]))
            Crop_ = np.hstack((Crop_, site['Crop'][~ierror]))
            LeafType_ = np.hstack((LeafType_, site['LeafType'][~ierror]))
            C3C4_ = np.hstack((C3C4_, site['C3C4'][~ierror]))
            LeafPhen_ = np.hstack((LeafPhen_, site['LeafPhen'][~ierror]))
            Koppen_ = np.hstack((Koppen_, site['Koppen'][~ierror]))

            Trait_value = traits[trait_name][~ierror]
            Trait_name = [trait_name for _ in range(len(Trait_value))]
            Trait_ID = [id_traits[trait_name] for _ in range(len(Trait_value))]

            TRYTraitName_ = np.hstack((TRYTraitName_, Trait_name)) # multiplied into list
            TRYTraitID_ = np.hstack((TRYTraitID_,Trait_ID))  #multiplied into list
            TRYTrait_value_ = np.hstack((TRYTrait_value_, Trait_value))

        # ierror = (Latitude_ < -90) + (Latitude_ > 90) + (Longitude_ < -180) + (Longitude_ > 180)
        # Latitude_ = Latitude_[~ierror]
        # Longitude_ = Longitude_[~ierror]
        # Plantgroup_ = Plantgroup_[~ierror]
        # Crop_ = Crop_[~ierror]
        # LeafType_ = LeafType_[~ierror]
        # C3C4_ = C3C4_[~ierror]
        # LeafPhen_ = LeafPhen_[~ierror]
        # PFT_id_ = PFT_id_
        # TRYTraitID_ = TRYTraitID_
        # TRYTraitName_ = TRYTraitName_
        # TRYTrait_value_ = TRYTrait_value_
        # TRYSpeciesID_ = TRYSpeciesID_
        # TRYSpeciesName_ = TRYSpeciesName_

        return Latitude_, Longitude_, Plantgroup_, Crop_, LeafType_, C3C4_, LeafPhen_, PFT_id, TRYTraitID_, TRYTraitName_, TRYTrait_value_, TRYSpeciesID_, TRYSpeciesName_

    def ReadTryFile(self):
        import csv

        Rows = []
        aux_data_provider = get_aux_data_provider()
        path_to_file = os.path.abspath(os.path.join(self.directory_data, 'Try_Database', 'PROSAILTraits.txt'))
        if not aux_data_provider.assure_element_provided(path_to_file):
            logging.warning(f'Could not find {path_to_file}')
        with open(path_to_file, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            # spamreader = csv.reader(csvfile)
            # reader = csv.DictReader(csvfile)
            for row in spamreader:
                # print ', '.join(row)
                Rows.append(row)
        Rows = np.array(Rows)

        Lat_ = Rows[1:, 18]
        Lon_ = Rows[1:, 19]

        Plantgroup_ = Rows[1:, 20]
        Crop_ = Rows[1:, 21]
        LeafType_ = Rows[1:, 22]
        C3C4_ = Rows[1:, 23]
        LeafPhen_ = Rows[1:, 24]
        PFT_id = np.arange(1, 18)


        id_traitid = [i for i, name in enumerate(Rows[0, :]) if (name == "TRYTraitID")]
        id_traitname = [i for i, name in enumerate(Rows[0, :]) if (name == "Trait_name")]
        id_traitvalue = [i for i, name in enumerate(Rows[0, :]) if (name == "NewValue")]
        id_speciesid = [i for i, name in enumerate(Rows[0, :]) if (name == "TRYSpec_nu")]
        id_speciesname = [i for i, name in enumerate(Rows[0, :]) if (name == "Spec_name")]


        TRYTraitID = Rows[1:, id_traitid]
        TRYTraitName = Rows[1:, id_traitname]
        trait_value = Rows[1:, id_traitvalue]
        TRYSpeciesID = Rows[1:, id_speciesid]
        TRYSpeciesName = Rows[1:, id_speciesname]

        return Lat_, Lon_, Plantgroup_, Crop_, LeafType_, C3C4_, LeafPhen_, PFT_id, TRYTraitID, TRYTraitName, trait_value, TRYSpeciesID, TRYSpeciesName

    def ReadMeteorologicalData(self, doystr):
        """
        Read Meteorological Variables
        This function is a placeholder to be used when the dynamic functionality is created.

        :param doystr: string containing date&time '2007-12-31 04:23' for processing needs to be performed
        :returns: Meteorological data (to be used for upscaling Peak Biomass traits to seasonal priors)
        :rtype:

        """
        MeteoData = None
        Meteo_lon = None
        Meteo_lat = None

        if self.plotoption == 2:
            plt.figure(figsize=[20, 20])
            plt.imshow(MeteoData)
            plt.title('Meteorological Data (missing at this moment)')

        return MeteoData, Meteo_lon, Meteo_lat

    # Offline processing
    def DownloadCrossWalkingTable(self):
        """
        Download Crosswalking table
        Here the Cross walking table is downloaded to create the CCI landcover map. At the moment this is simply
        a placeholder for future functionality.

        :returns: -
        :rtype:

        """
        # According to Pulter et al, Plant Functional classification for
        # earth system models: resuls from the European Space
        # Agency's land Cover Climate Change Initiative, 2015,
        # Geosci Model Dev., 8, 2315-2328, 2015.

        link2LCC_map = ('https://storage.googleapis.com/cci-lc-v207/ESACCI-LC-'
                        'L4-LCCS-Map-300m-P1Y-2015-v2.0.7.nc.zip')
        link2CrossWalkingtable = ('http://maps.elie.ucl.ac.be/CCI/viewer/'
                                  'download/lc-user-tools-3.14.zip')

    def RunCrossWalkingTable(self, Path2CWT_tool=None, Path2LC=None):
        """
        Creating CCI landcover maps (using crosswalking table).

        please note that to run the crosswalking tool, the specific requirements for BEAM need
        to be met (java64bit + ...)

        :param Path2CWT_tool:
        :param Path2LC:
        :returns: -
        :rtype:

        """

        if Path2CWT_tool is None:
            Tooldir = '~/Data/Prior_Engine/Tool/lc-user-tools-3.14/'
            Path2CWT_tool = Tooldir + 'bin/remap.sh'
            Path2CWT_file = Tooldir + 'resources/Default_LCCS2PFT_LUT.csv'
        if Path2LC is None:
            Path2LC = ('~/Data/Prior_engine/Data/LCC/'
                       'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.nc')

        string2execute = (Path2CWT_tool + ' -PuserPFTConversionTable=' +
                          Path2CWT_file + ' ' + Path2LC)
        os.system(string2execute)

        # Please note that we use the default crosswalking table over here.
        # This does not distinguish between C3/C4 crops/grasses, or identify
        # non-vascular plants. For this we need to acquire the
        # most recent cross-walking table used Druel. A et al, Towards a more
        # detailed representation of high-latitutde vegetation
        # in the global land surface model ORCHIDEE (ORC-HL-VEGv1.0).

        # Altneratively we can use the Synmap c3/c4 fraction map. to
        # distinguish make this distinction for grasses.
        return

    def CreateDummyDatabase(self):
        """
        create netcdf Database files to hold database values

        :returns: -
        :rtype:

        """

        # define variables
        varnames = SUPPORTED_VARIABLES
        descriptions = ['Effective Leaf Area Index',
                        'Leaf Chlorophyll Content', 'Leaf Senescent material',
                        'Leaf Carotonoid Content', 'Leaf Water Content',
                        'Leaf Dry Mass', 'Structural Parameter',
                        'Average Leaf Angle', 'hotspot parameter',
                        'Soil Brightness parameter', 'Soil Wetness parameter']
        units = ['m2/m2', 'ug/cm2', '-', 'ug/cm2', 'cm', 'g/cm2', '-',
                 'degrees', '-', '-', '-']

        # create netcdf file to hold database values
        dataset = Dataset(self.path2Trait_file, 'w', format='NETCDF4')

        # setup Netcdf file
        dataset.description = 'Database for Prior-Engine'
        dataset.history = 'Created' + time.ctime(time.time())
        dataset.source = 'Data obtained from TRY-database'

        # Define dimensions
        Npft = 16
        Nspecies = 100
        pftdim = dataset.createDimension('pft', Npft)
        speciesdim = dataset.createDimension('type', Nspecies)
        occurrencedim = dataset.createDimension('occ', None)

        # create variables in dataset
        occurrence = dataset.createVariable('Occurrences', np.float32, ('occ'))
        occurrence.description = 'Unique PFT/Species entry into database'

        pft = dataset.createVariable('PFTs', np.float32, ('pft'))
        pft.description = 'PFT classification according to ORCHIDEE'

        species = dataset.createVariable('Species', np.float32, ('type'))
        species.description = 'Species name according to ??'

        for ivar, varname in enumerate(varnames):
            var = dataset.createVariable(varname, np.float32,
                                         ('occ', 'pft', 'type'), zlib=True)
            var.units = units[ivar]
            var.description = descriptions[ivar]

            # Fill file with random variables
            Nvar = 10  # np.random.randint(1,10)
            var[0:Nvar, :, :] = np.random.uniform(size=(Nvar, Npft, Nspecies))

        dataset.close()
        os.system('chmod 755 "' + self.path2Trait_file + '"')

    def CreateRealDatabase(self):

        import csv
        # Lat2_, Lon2_, Plantgroup2_, Crop2_, LeafType2_, C3C42_, LeafPhen2_, PFT_id2, TRYTraitID2, TRYTraitName2, trait_value2, TRYSpeciesID2, TRYSpeciesName2= self.ReadTryFile()
        # trait_value2 = np.array([v.astype(float) for v in trait_value2[:, 0]])

        Lat_, Lon_, Plantgroup_, Crop_, LeafType_, C3C4_, LeafPhen_, PFT_id, TRYTraitID, TRYTraitName, trait_value, TRYSpeciesID, TRYSpeciesName = self.ReadTryDatabase()

        ID = self.ExtractPFT4TryDatabaseEntries(Lat_, Lon_, Plantgroup_, Crop_, LeafType_, C3C4_, LeafPhen_)


        # LUT to link ID's to Name's
        TRYTraitID_u = np.unique(TRYTraitID)
        id_trait = dict()
        id_trait['cdm'] = '11'
        id_trait['ala'] = '3'
        id_trait['cab'] = '413'
        id_trait['car'] = '491'
        id_trait['cwc'] = '999001'
        id_trait['cw'] = '999001'
        id_trait['N'] = '999002'

        TRYSpeciesID_u = np.unique(TRYSpeciesID)
        # id_speciesname = [TRYSpeciesName[ids == TRYSpeciesID][0] for ids in TRYSpeciesID_u]

        #########################################################################################################
        # Define file-setup
        #########################################################################################################

        # define variables
        varnames = ['cab', 'cb', 'car', 'cw', 'cdm', 'N', 'ala', 'h', 'bsoil', 'psoil']#, 'lai',
        # varnames = ['cab', 'car', 'cw', 'N', 'ala','cdm']  # , 'lai','cdm',
        # varnames = ['cab', 'car' ,'N']
        # varnames = ['N','car'] work well together in 1 file
        # varnames = ['cab','cdm']#,'cab'
        descriptions = ['Leaf Chlorophyll Content', 'Leaf Senescent material', \
                        'Leaf Carotonoid Content', 'Leaf Water Content', 'Leaf Dry Mass', \
                        'Structural Parameter', 'Average Leaf Angle', 'hotspot parameter', \
                        'Soil Brightness parameter', 'Soil Wetness parameter'] #'Effective Leaf Area Index',
        units = ['ug/cm2', '-', 'ug/cm2', 'cm', 'g/cm2', '-', 'degrees', '-', '-', '-']    #'m2/m2',         # to be replaced by directly extracting it from the ascii file?

        #########################################################################################################
        # Create file
        #########################################################################################################
        #Create netcdf file to hold database values
        dataset = Dataset(self.path2Trait_file, 'w', format='NETCDF4')

        # setup Netcdf file
        dataset.description = 'Database for Prior-Engine'
        dataset.history = 'Created' + time.ctime(time.time())
        dataset.source = 'Data obtained from TRY-database'

        # Define dimensions
        Ntraits = len(id_trait)

        Nspecies = len(TRYSpeciesID_u)
        NPFT     = len(PFT_id)

        pftdim = dataset.createDimension('pft', NPFT)
        speciesdim = dataset.createDimension('type', Nspecies)
        occurrencedim = dataset.createDimension('occ', None)

        # create variables in dataset
        occurrence = dataset.createVariable('Occurrences', np.float32, ('occ'))
        occurrence.description = 'Unique PFT/Species entry into database'

        pft = dataset.createVariable('PFTs', np.float32, ('pft'))
        pft.description = 'PFT classification according to ORCHIDEE'

        species = dataset.createVariable('Species', np.float32, ('type'))
        species.description = 'Species name according to ??'

        # fill 3D database with Trait values
        Ivalid = (~np.isnan(trait_value))*(trait_value>=0)

        Nmax = 0
        var = dict()
        for ivar, varname in enumerate(varnames):
            if varname in id_trait:
                # I1 = (id_trait[varname] == TRYTraitID2[:, 0]) * Ivalid
                I1 = (id_trait[varname] == TRYTraitID) * Ivalid

                var[varname] = dataset.createVariable(varname, np.float32, ('occ', 'pft', 'type'), zlib=True)
                var[varname].units = units[ivar]
                var[varname].description = descriptions[ivar]
                for ipft in range(NPFT):
                    I2 = (PFT_id[ipft] == ID)  # * (TRYSpeciesID_u[ispecies] == TRYSpeciesID[:,0])

                    # print('ivar = %7.0f' % ivar + 'ipft = %7.0f' % ipft ) # + ',%7.0f' % ispecies
                    Nocctot = 0
                    if np.any(I1 * I2):

                        for ispecies in range(Nspecies):
                            # for ispecies in range(100):
                            # I3 = (TRYSpeciesID_u[ispecies] == TRYSpeciesID2[:,0])
                            I3 = (TRYSpeciesID_u[ispecies] == TRYSpeciesID[:])
                            Itot = I1 * I2 * I3

                            if np.any(Itot):
                                trait_value_selected = trait_value[Itot]

                                # cdm is some cases were provided in different units. this is to correct that
                                if varname is 'cdm':
                                    iuniterror = trait_value_selected > 1.e-1
                                    trait_value_selected[iuniterror] = trait_value_selected[iuniterror] / 1.e4

                                Noccurrence = len(trait_value_selected)# * (~np.all(np.isnan(trait_value_selected)))
                                Nmax  = np.max([Noccurrence,Nmax])
                                # print(trait_value_selected)

                                # Fill file
                                Nocctot = Nocctot+Noccurrence
                                # if Noccurrence>0:
                                var[varname][0:Noccurrence, ipft, ispecies] = trait_value_selected[0:Noccurrence]
                                # var[varname][500, ipft, ispecies] = -999
                                # if Noccurrence>100:
                                # print(trait_value_selected[0:Noccurrence])
                                    # var[Noccurrence:, ipft, ispecies] = -999.
                                # else:
                                #     var[:, ipft, ispecies] = -999


                            # else:
                            #     var[:, ipft, ispecies] = -998.
                    # else:
                    #     var[:, ipft, :] = -997.
                    # else:
                    #     var[:, :, :] = -999.

                    print('Number of species/occ-entries (for '  + varname + ' and PFTid %02.0f' % ipft + ') =  %03.0f'  % Nocctot)

        # fill in last column with nan values to ensure proper creation of all matrices
        for varname in var:
            for ipft in range(NPFT):
                for ispecies in range(Nspecies):
                    var[varname][Nmax, ipft, ispecies] = np.NaN

        dataset.close()
        os.system('chmod 755 "' + self.path2Trait_file + '"')

    # Static Processing data functions
    def RescaleCLM(self, CLM_lon, CLM_lat, CLM_map, LCC_lon, LCC_lat):
        """
        Collocate Climate Zone map with landcover coordinates
        The Climate Zone map has a different resolution/grid than the Landcover map. This preprocessing is performed
        to collocate both (in order to facilitate the merging downstream.)

        :param CLM_lon: array containing the longitude values of the Climate Zone map
        :param CLM_lat: array containing the latitude values of the Climate Zone map
        :param CLM_map: array containing the Climate Zone map
        :param LCC_lon: array containing the longitude values of the CCI Landcover map
        :param LCC_lat: array containing the latitude values of the CCI Landcover map
        :returns: array containing the Regridded Climate Zone map
        :rtype:

        """
        x, y = np.meshgrid(CLM_lon, CLM_lat)
        n = x.size

        F = RegularGridInterpolator.NearestNDInterpolator(
            (np.resize(x, n), np.resize(y, n)), np.resize(CLM_map, n))

        loni, lati = np.meshgrid(LCC_lon, LCC_lat)
        CLM_map_i = F(loni, lati)

        if self.plotoption == 4:
            plt.figure(figsize=[20, 20])
            plt.subplot(2, 1, 1)
            plt.imshow(CLM_map)
            plt.subplot(2, 1, 2)
            plt.imshow(CLM_map_i[0::2, 0::2])

        return CLM_map_i

    def Combine2PFT(self, LCC_map, CLM_map_i):
        """
        Create PFT maps using CCI Landcover and Koppen Climate zone information
        :param LCC_map: CCI Landcover map
        :param CLM_map_i: Regridded Koppen Climate Zone map
        :returns: PFT occurrence map, PFT classes, Number of PFTs, PFT ids
        :rtype:

        """
        iwater = (CLM_map_i == 0)
        itropical = (CLM_map_i >= 1) * (CLM_map_i <= 7)
        itemporate = (CLM_map_i >= 8) * (CLM_map_i <= 16)
        iboreal = (CLM_map_i >= 17) * (CLM_map_i <= 28)
        ipolar = (CLM_map_i >= 29) * (CLM_map_i <= 32)

        Nlon = np.shape(CLM_map_i)[0]
        Nlat = np.shape(CLM_map_i)[1]
        Npft = 16

        PFT = np.zeros([Nlon, Nlat, Npft])
        # water
        PFT[:, :, 0] = (iwater + (LCC_map['Water'] > 0)
                        + (LCC_map['Snow_Ice'] > 0))
        # Trees: Tropical: Broadleaf: Evergreen
        PFT[:, :, 1] = itropical * LCC_map['Tree_Broadleaf_Evergreen']
        # Trees: Tropical: Broadleaf: Raingreen
        PFT[:, :, 2] = itropical * LCC_map['Tree_Broadleaf_Deciduous']

        # Trees: Temperate: Needleleaf: Evergreen
        PFT[:, :, 3] = itemporate * LCC_map['Tree_Needleleaf_Evergreen']
        # Trees: Temperate: Broadleaf: Evergreen
        PFT[:, :, 4] = itemporate * LCC_map['Tree_Broadleaf_Evergreen']
        # Trees: Temperate: Broadleaf: Summergreen
        PFT[:, :, 5] = itemporate * LCC_map['Tree_Broadleaf_Deciduous']

        # Trees: Boreal: Needleleaf: Evergreen
        PFT[:, :, 6] = iboreal * LCC_map['Tree_Needleleaf_Evergreen']
        # Trees: Boreal: Broadleaf: Summergreen
        PFT[:, :, 7] = iboreal * LCC_map['Tree_Broadleaf_Deciduous']
        # Trees: Boreal: Needleleaf: Summergreen
        PFT[:, :, 8] = iboreal * LCC_map['Tree_Broadleaf_Deciduous']

        # Grasses: Natural: C3: Global
        PFT[:, :, 9] = (itemporate + iboreal) * LCC_map['Natural_Grass']
        # Grasses: Natural: C3: Arctic
        PFT[:, :, 10] = ipolar * LCC_map['Natural_Grass']
        # Grasses: Natural: C4:
        PFT[:, :, 11] = itropical * LCC_map['Natural_Grass']

        # 12  Crops: C3
        PFT[:, :, 12] = (itemporate + iboreal) * LCC_map['Managed_Grass']
        # 13  Crops: C4
        PFT[:, :, 13] = ~(itemporate + iboreal) * LCC_map['Managed_Grass']
        # 14  Shrubs
        PFT[:, :, 14] = LCC_map['Shrub_Broadleaf_Evergreen'] + \
                        LCC_map['Shrub_Broadleaf_Deciduous'] + \
                        LCC_map['Shrub_Needleleaf_Evergreen'] + \
                        LCC_map['Shrub_Needleleaf_Deciduous']

        # not accounted for in this version of the LCC.For this the
        # cross-walking table should be modified.This should be handled
        # in the beta version

        # 15  Non-Vascular (mosses)
        PFT[:, :, 15] = LCC_map['Shrub_Needleleaf_Deciduous'] * 0
        PFT_classes = ['water',
                       'Trees: Tropical: Broadleaf: Evergreen',
                       'Trees: Tropical: Broadleaf: Raingreen',
                       'Trees: Temperate: Needleleaf: Evergreen',
                       'Trees: Temperate: Broadleaf: Evergreen',
                       'Trees: Temperate: Broadleaf: Summergreen',
                       'Trees: Boreal: Needleleaf: Evergreen',
                       'Trees: Boreal: Broadleaf: Summergreen',
                       'Trees: Boreal: Needleleaf: Summergreen',
                       'Grasses: Natural: C3: Global',
                       'Grasses: Natural: C3: Arctic',
                       'Grasses: Natural: C4:',
                       'Crops: C3',
                       'Crops: C4',
                       'Shrubs',
                       'Non-Vascular']
        Npft = len(PFT_classes)
        PFT_ids = np.arange(0, Npft)

        # visualize
        if self.plotoption == 5:
            Nc = 4
            Npft = np.shape(PFT)[2]
            Nr = int(np.ceil(Npft / Nc + 1))

            plt.figure(figsize=[20, 20])
            for i in range(Npft):
                plt.subplot(Nr, Nc, i + 1)
                plt.imshow(PFT[:, :, i])
                plt.colorbar()
                plt.title(PFT_classes[i])

        return PFT, PFT_classes, Npft, PFT_ids

    def AssignPFTTraits2Map(self, PFT, PFT_ids, varnames):
        """
        Create Vegetation trait Prior map, using the Trait-database and PFT distribution maps
        This function sets up a parallel processing chain around
        - processespercore: here the actual assignment of traits to PFT distributions is performed

        :param PFT: arrays containing global Maps of PFT distributions
        :param PFT_ids: a list containing PFT ids
        :param varnames: list of variables to be converted into global file
        :returns: map of vegetation trait-averages per PFT id, map of vegetation trait-uncertainties per PFT id
        :rtype:

        """
        # from multiprocessing import Pool
        from functools import partial

        # open Trait data
        TRAITS_ttf_avg = dict()
        TRAITS_ttf_unc = dict()

        # Process in parallel
        option_parallel = 0
        if option_parallel == 1:
            pr = partial(processespercore, PFT=PFT, PFT_ids=PFT_ids, VegetationPriorCreator=self)
            ret = parmap(pr, varnames, nprocs=3)
            for ivar, varname in enumerate(varnames):
                TRAITS_ttf_avg[varname] = ret[ivar][0][:, :]
                TRAITS_ttf_unc[varname] = ret[ivar][1][:, :]
        else:
            for varname in varnames:
                TRAIT_ttf_avg, TRAIT_ttf_unc            =   processespercore(varname, PFT, PFT_ids, self)

                # write back to output
                TRAITS_ttf_avg[varname]                 =   TRAIT_ttf_avg
                TRAITS_ttf_unc[varname]                 =   TRAIT_ttf_unc


        if self.plotoption == 6:
            Nc = 4.
            Nvar = len(varnames)
            Nr = int(np.ceil(Nvar / Nc))
            plt.figure(figsize=[20, 20])
            for i, varname in enumerate(varnames):
                plt.subplot(Nr, Nc, i + 1)
                plt.imshow(TRAITS_ttf_avg[varname])
                plt.colorbar()
                plt.title(varname)

            plt.figure(figsize=[20, 20])
            for i, varname in enumerate(varnames):
                plt.subplot(Nr, Nc, i + 1)
                plt.imshow(TRAITS_ttf_unc[varname])
                plt.colorbar()
                plt.title(varname + '_unc')

        return TRAITS_ttf_avg, TRAITS_ttf_unc

    # Dynamical Processing data functions
    def PhenologicalEvolution(self, Prior_pbm_avg, Prior_pbm_unc,
                              doystr, Meteo_map_i=None):
        """
        Model the Phenological Evolution of Vegetation traits
        This function is a placeholder to be used when the dynamic functionality is created.

        :param Prior_pbm_avg: Vegetation trait-averages at Peak Biomass
        :param Prior_pbm_unc: Vegetation trait-uncertainty (@PBM)
        :param doystr: string containing date&time '2007-12-31 04:23' for processing needs to be performed
        :param Meteo_map_i: Place_holder for meteorological data-files
        :returns: Temporal Prior-averages, Temporal Prior-uncertainties
        :rtype:

        """
        if Meteo_map_i is None:
            Prior_avg = Prior_pbm_avg
            Prior_unc = Prior_pbm_unc
        else:
            print("Here we are going to do some interesting bit with"
                  " phenological models at %s" % (doystr))

        return Prior_avg, Prior_unc


    # Writing data functions
    def WriteOutput(self, LCC_lon, LCC_lat, Prior_avg, Prior_unc,
                    doystr='static'):
        """
        Write Vegetation Prior data (mean/unc) to NETCDF outputfiles. This functionality is obsolete as all outputs
        are written to GeoTiff files

        :param LCC_lon: longitude of the Prior data (same as used Landcover map)
        :param LCC_lat: latitude of the Prior data (same as used Landcover map)
        :param Prior_avg: Vegetation prior average values
        :param Prior_unc: Vegetation prior uncertainty values
        :param doystr: string containing date&time '2007-12-31 04:23' for data to be written
        :returns: -
        :rtype:

        """
        # varnames = [name for name in Prior_avg]
        # latstr = ('[%02.0f' % self.lat_study[0]
        #           + ' %02.0fN]' % self.lat_study[1])
        # lonstr = ('[%03.0f' % self.lon_study[0]
        #           + ' %03.0fE]' % self.lon_study[1])
        # filename = (self.path2Traitmap_file[:-3] + '_' + doystr + '_' + latstr
        #             + '_' + lonstr + '.nc')
        #
        # dataset = Dataset(filename, 'w', format='NETCDF4_CLASSIC')
        # dataset.description = 'Transformed Priors at doy: ' + doystr
        # dataset.history = 'Created' + time.ctime(time.time())
        #
        # dataset.cdm_data_type = 'grid'
        # dataset.contact = 'j.timmermans@cml.leidenuniv.nl'
        # dataset.Conventions = 'CF-1.6'
        #
        # dataset.creator_email = 'j.timmermans@cml.leidenuniv.nl'
        # dataset.creator_name = 'Leiden University'
        # dataset.date_created = '20180330T210326Z'
        # dataset.geospatial_lat_max = '90.0'
        # dataset.geospatial_lat_min = '-90.0'
        # dataset.geospatial_lat_resolution = '0.002778'
        # dataset.geospatial_lat_units = 'degrees_north'
        # dataset.geospatial_lon_max = '180.0'
        # dataset.geospatial_lon_min = '-180.0'
        # dataset.geospatial_lon_resolution = '0.002778'
        # dataset.geospatial_lon_units = 'degrees_east'
        # dataset.id = 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7'
        # dataset.institution = 'Universite catholique de Louvain'
        # dataset.keywords = 'land cover classification,satellite,observation'
        # dataset.keywords_vocabulary = ('NASA Global Change Master Directory (GCMD) Science Keywords')
        # dataset.license = 'ESA CCI Data Policy: free and open access'
        # dataset.naming_authority = 'org.esa-cci'
        # dataset.product_version = '2.0.7'
        # dataset.project = 'Climate Change Initiative - European Space Agency'
        # dataset.references = 'http://www.esa-landcover-cci.org/'
        # dataset.source = ('MERIS FR L1B version 5.05, MERIS RR L1B version 8.0, SPOT VGT P')
        # dataset.spatial_resolution = '300m'
        # dataset.standard_name_vocabulary = ('NetCDF Climate and Forecast (CF) Standard Names version 21')
        # dataset.summary = ('This dataset contains the global ESA CCI land cover classification map derived from satellite data of one epoch.')
        # dataset.TileSize = '2048:2048'
        # dataset.time_coverage_duration = 'P1Y'
        # dataset.time_coverage_end = '20151231'
        # dataset.time_coverage_resolution = 'P1Y'
        # dataset.time_coverage_start = '20150101'
        # dataset.title = 'ESA CCI Land Cover Map'
        # dataset.tracking_id = '202be995-43d8-4e3a-9607-1bd3f02a925e'
        # dataset.type = 'ESACCI-LC-L4-LCCS-Map-300m-P1Y'
        #
        # # create dimensions
        # Nlon = len(LCC_lon)
        # Nlat = len(LCC_lat)
        #
        # londim = dataset.createDimension('lon', Nlon)
        # latdim = dataset.createDimension('lat', Nlat)
        #
        # # create variables in dataset
        # lon = dataset.createVariable('lon', np.float32, ('lon'))
        # lon.description = 'Longitude'
        # lon.units = 'degrees_east'
        # lon[:] = LCC_lon
        #
        # lat = dataset.createVariable('lat', np.float32, ('lat'))
        # lat.description = 'Latitude'
        # lat.units = 'degrees_north'
        # lat[:] = LCC_lat
        #
        # for varname in varnames:
        #     var = dataset.createVariable(varname, np.float32, ('lon', 'lat'), zlib=True)
        #     var.units = ''
        #     var[:] = Prior_avg[varname]
        #     var_unc = dataset.createVariable(varname + '_unc',np.float32, ('lon', 'lat'))
        #     var_unc.units = ''
        #     var_unc[:] = Prior_unc[varname]
        #
        # dataset.close()
        # os.system('chmod 755 "' + filename + '"')
        # print('%s', filename)
        return

    def WriteGeoTiff(self, LCC_lon, LCC_lat, Prior_avg,
                     Prior_unc, doystr='static'):
        """
        Write Vegetation Prior data (mean/unc) to GEOTIFF outputfiles.
        :param LCC_lon: longitude of the Prior data (same as used Landcover map)
        :param LCC_lat: latitude of the Prior data (same as used Landcover map)
        :param Prior_avg: Vegetation prior average values
        :param Prior_unc: Vegetation prior uncertainty values
        :param doystr: string containing date&time '2007-12-31 04:23' for data to be written
        :returns: -


        """
        Nlayers = 2
        latstr = ('[%02.0f' % self.lat_study[0]
                  + '_%02.0fN]' % self.lat_study[1])
        lonstr = ('[%03.0f' % self.lon_study[0]
                  + '_%03.0fE]' % self.lon_study[1])


        varnames = [name for name in Prior_avg]
        drv = gdal.GetDriverByName("GTIFF")
        for i, varname in enumerate(varnames):
            filename = (self.path2Traitmap_file[:-3] + '_' + varname + '_'
                        + doystr + '_' + latstr + '_' + lonstr + '.tiff')
            dst_ds = drv.Create(filename, np.shape(LCC_lon)[0],
                                np.shape(LCC_lat)[0], Nlayers,
                                gdal.GDT_Float32,
                                options=["COMPRESS=LZW", "INTERLEAVE=BAND",
                                         "TILED=YES"])
            resx = LCC_lon[1] - LCC_lon[0]
            resy = LCC_lat[1] - LCC_lat[0]

            dst_ds.SetGeoTransform([min(LCC_lon), resx, 0, max(LCC_lat),
                                    0, -np.abs(resy)])
            dst_ds.SetProjection(
                'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS84",6378137,'
                '298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY'
                '["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",'
                '0.0174532925199433],AUTHORITY["EPSG","4326"]]')

            dst_ds.GetRasterBand(1).WriteArray(Prior_avg[varname])
            dst_ds.GetRasterBand(1).SetDescription(varname + '-mean')
            dst_ds.GetRasterBand(2).WriteArray(Prior_unc[varname])
            dst_ds.GetRasterBand(2).SetDescription(varname + '-unc')
            dst_ds = None

    def CombineTiles2Virtualfile(self, variable, doystr, directory_data):
        """
        Combine all geotiff files into a virtual global file

        :param variable: variable to be converted into global file
        :param doystr: string containing date&time '2007-12-31 04:23' for which global file needs to be created
        :returns: the filename of the global VRT file
        :rtype:

        """
        aux_data_provider = get_aux_data_provider()
        dir = directory_data + 'Priors/'
        file_name = 'Priors_' + variable + '_' + doystr + '_global.vrt'
        # todo exchange 125 in upcoming versions with doy
        # list_of_files = glob.glob(dir + 'Priors*_' + variable + '_*125*.tiff')
        list_of_files = aux_data_provider.list_elements(dir, f'Priors*_{variable}_*125*.tiff')
        if len(list_of_files) == 0:
            raise UserWarning('No input files found for variable {}'.format(variable))

        list_of_files_as_strings = []
        for filename in list_of_files:
            if aux_data_provider.assure_element_provided(filename):
                list_of_files_as_strings.append('"' + filename + '"')

        files = " ".join(list_of_files_as_strings)
        output_file_name = '{}{}'.format(self.output_directory, file_name)
        os.system('gdalbuildvrt -te -180 -90 180 90 ' + output_file_name + ' ' + files)
        return output_file_name

    def compute_prior_file(self):
        """
        Combine Tiles into single Prior VRT file

        :returns: filename of specific VRT file
        :rtype:

        """
        # Define variables
        if self.variable is None:
            self.variables = SUPPORTED_VARIABLES

        # time = parse(self.datestr)
        time = self.date
        doystr = time.strftime('%j')

        if self.ptype == 'database':
            # 0. Setup Processing
            # filenames = self.CombineTiles2Virtualfile(variables, doystr)
            filenames = self.CombineTiles2Virtualfile(self.variable, doystr, self.directory_data)

        elif self.ptype == 'user':
            user_directory = '/'.join(self.output_directory.split('/')[:-2]) + '/user/'
            self.path2Traitmap_file = user_directory + 'Priors/Priors.nc'

            if  not os.path.exists(user_directory + 'Priors/'):
                 os.mkdir(user_directory)
                 os.mkdir(user_directory + 'Priors/')

            self.ProcessData(variables = self.variable)
            filenames = self.CombineTiles2Virtualfile(self.variable, doystr, user_directory)

        return filenames

if __name__ == "__main__":
    from multiply_prior_engine import PriorEngine
    import datetime

    config = _get_config('config2.yaml')

    option_recreate_priors = 1
    # vars = ['cbrown','bsoil','psoil']
    # vars = ['car', 'cw', 'N']
    vars = ['cdm','cab','ala']

    VegPrior = VegetationPriorCreator(var=vars,
                                      datestr='2007-12-31 04:23',
                                      ptype='database',
                                      config=config)
    datestr = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d%H%M%S')
    VegPrior.path2Trait_file = VegPrior.directory_data + 'Trait_Database/' + 'Traits_'+datestr+'.nc'
    VegPrior.CreateRealDatabase()

    if option_recreate_priors:
        print('Reperform calculations')

        # use a different prior for cdm, instead of the 'try database' as there were errors in the database
        # VegPrior.mu['cdm'] = 0.003

        # Define study area as global instead of through the config file
        VegPrior.lon_study = [-180, 180]
        VegPrior.lat_study = [-90, 90]

        # create Initial data-files to be retrieved from prior-engine
        VegPrior.ProcessData(variables=vars)
    else:
        print('using earlier calculations')

    ####################################################################
    # VegPrior.RetrievePrior(variables=['lai','cab'],datestr='2007-12-31 04:23', ptype='database')





