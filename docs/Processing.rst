Processing Flow
=================

Priors are provided by the MULTIPLY prior engine for the respective forward operators. The relationships are shown in following figure:

.. figure:: img/prior_forward.png
   :alt: prior to forward operator relationship

   Figure 1: Relationship of priors to their respective forward operators.

.. note::
   For information on **user defined prior files** please see the section on `Usage <Usage.rst>`_.

Description of Prior Generation
-----------------------------------
This prototype is capable of delivering for both vegetation priors as well as soil priors spanning all variables required in the forward operators. The overall processing chain is divided up to two parts (dealing with the soil prior and the vegetation prior).

The optical prior engine is designed to deliver prior information to the inference engine specifically for the leaves and vegetation. The overall flow of the prior-engine is illustrated by Figure 2.

The 'microwave' prior engine is designed to deliver prior information for soil parameters. The overall flow of this part of the prior-engine is illustrated by Figure 3.

In these flowcharts a distinction is made between the current implementation of the prototype (green) and the final foreseen version of the prior engine (red).
In order for completeness a place-holder (orange) process is embedded into the flowchart. In addition, in the final version of the prior engine the users themselves can choose between how the specific prior are used (see `Usage <Usage.rst>`_). User-selections are obtained from the configuration-file with which the MULTIPLY framework is run. This is represented in the flowchart by orange selection boxes.
Prior data specified by the User is currently not visualized for every prior generator.

Vegetation Priors
++++++++++++++++++
Within the prototype version of the module, the values of the priors are consistent with @peak biomass; no dynamical component is integrated into the prototype module.

.. figure:: img/flow_prior_opt.png
   :alt: flow of 'optical' prior engine

   Figure 2: Flow in 'optical' prior engine


Soil Priors
+++++++++++++

The included priors for soil moisture are currently twofold:

1) a climatological prior based on `ESA CCI SM v04.4 <https://esa-soilmoisture-cci.org/>`_ data
2) a dynamic prior based on `SMAP <https://smap.jpl.nasa.gov/data/>`_ data

Please see the overall flow of this prior creator sub-engine below:

.. figure:: img/flow_prior_mic.png
   :alt: flow of 'microwave' prior engine

   Figure 3: Flow in 'microwave' prior engine

Climatologic Priors
********************

Mattia et al. [Mattia]_ show that the usage of climatological mean soil moisture information significantly improves soil moisture estimates from active microwave observations. Therefore, a soil moisture climatology is used as prior to get a general idea of the amplitude, variability and seasonal behaviour of the in situ soil moisture. Furthermore, a dynamic daily coarse resolution product is consulted for an a priori estimation of the current state.

The climatological prior data set has been generated from the global ESA CCI SM v04.4 COMBINED product which is derived from a combination of active and passive satellite sensors over the period 1978 - 2018 [GRUBER2019]_. Originally, the data set provides daily surface soil moisture with a spatial resolution of 0.25 degree ([Dorigo]_; [Gruber]_; [Liu]_). The data was aggregated to monthly means. Uncertainty is given by the intra-monthly standard deviation.
There is also a interpolation routine included to allow for smooth inter monthly transitions.


.. figure:: img/clim_point.png
   :alt: Climatology soil moisture (bars) at point-scale with interpolated values (line)

   Figure 4: Climatology soil moisture (bars) at point-scale with interpolated values (line)

.. figure:: img/Clim_SM_4.png
   :alt: Exemplary climatological soil moisture prior (mean) for April

   Figure 5: Exemplary climatological soil moisture prior (mean) for April

Dynamic Priors
**************

Data from the Soil Moisture Active Passive (SMAP) project is used as dynamic prior ([Reichle]_). Specifically, the model-derived value-added Level 4 data product with 3-hourly estimates of soil moisture and respective error estimates at a 9 km resolution are averaged to daily values as the MULTIPLY platform assimilates data at this temporal resolution.

   "SMAP measurements provide direct sensing of soil moisture in the top 5 cm of the soil column. However, several of the key applications targeted by SMAP require knowledge of root zone soil moisture in the top 1 m of the soil column, which is not directly measured by SMAP. As part of its baseline mission, the SMAP project will produce model-derived value-added Level 4 data products to fill this gap and provide estimates of root zone soil moisture that are informed by and consistent with SMAP surface observations. Such estimates are obtained by merging SMAP observations with estimates from a land surface model in a data assimilation system. The land surface model component of the assimilation system is driven with observations-based meteorological forcing data, including precipitation, which is the most important driver for soil moisture. The model also encapsulates knowledge of key land surface processes, including the vertical transfer of soil moisture between the surface and root zone reservoirs. Finally, the model interpolates and extrapolates SMAP observations in time and in space, producing 3-hourly estimates of soil moisture at a 9 km resolution. The SMAP L4_SM product thus provides a comprehensive and consistent picture of land surface hydrological conditions based on SMAP observations and complementary information from a variety of sources." [JPL]_

The prior engine will rely on the `MULTIPLY data-access component <https://github.com/multiply-org/data-access>`_ to download the appropriate data sets. These are then converted to be used by the inference engine.
A valid registration on NASA's `Earthdata Service <https://urs.earthdata.nasa.gov/>`_ is necessary.


Technical Description
-----------------------

The processing chain in the prior engine is defined in a config file.
For now this looks like:

.. literalinclude:: ../multiply_prior_engine/sample_config_prior.yml


Internal Flow
++++++++++++++++

The internal flow and relations can be seen in figure 4.

.. figure:: img/PriorEngine.svg
   :alt: prior engine

   Figure 6: Prior Engine relationships


References
-------------

.. [GRUBER2019] Gruber, A., Scanlon, T., van der Schalie, R., Wagner, W., and Dorigo, W.: Evolution of the ESA CCI Soil Moisture climate data records and their underlying merging methodology, Earth Syst. Sci. Data, 11, 717-739, https://doi.org/10.5194/essd-11-717-2019, 2019

.. [JPL] https://smap.jpl.nasa.gov/data/

.. [Mattia] Mattia, F. et al. (2006) Using a priori information to improve soil moisture retrieval from ENVISAT ASAR AP data in semiarid regions. IEEE Trans. Geosci. Remote Sens. 44: 900–912.

.. [Dorigo] Dorigo, W. A., et al., 2017, ESA CCI Soil Moisture for improved Earth system understanding: State-of-the art and future directions, Remote Sensing of Environment, 203, 185-215, 2017, doi:10.1016/j.rse.2017.07.001.

.. [Gruber] Gruber, A., et al., 2017, Triple Collocation-Based Merging of Satellite Soil Moisture Retrievals, Transactions on Geoscience and Remote Sensing, 55(12), 1-13. doi:10.1109/TGRS.2017.2734070.

.. [Liu] Liu, Y. Y., et al., 2012, Trend-preserving blending of passive and active microwave soil moisture retrievals, Remote Sensing of Environment, 123, 280-297.

.. [Reichle] Reichle, R. et al. 2014. SMAP Algorithm Theoretical Basis Document: L4 Surface and Root-Zone Soil Moisture Product. SMAP Project, JPL D-66483, Jet Propulsion Laboratory, Pasadena, CA, USA.
