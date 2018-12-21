Prior Data
=============



Vegetation Prior Data
----------------------


.. note::
   TBD





Soil Moisture Prior Data
--------------------------

The provided prior data for the soil moisture domain is twofold. Mattia et al. (2006) show that the usage of climatological mean soil moisture information significantly improves soil moisture estimates from active microwave observations. Therefore, a soil moisture climatology is used as prior to get a general idea of the amplitude, variability and seasonal behaviour of the in situ soil moisture. Furthermore, a dynamic daily coarse resolution product is consulted for an a priori estimation of the current state.

The climatological prior data set has been generated from the global ESA CCI SM v04.4 COMBINED product which is derived from a combination of active and passive satellite sensors over the period 1978 - 2018. Originally, the data set provides daily surface soil moisture with a spatial resolution of 0.25 degree (Dorigo et al., 2017; Gruber et al. 2017; Liu et al. 2012). The data was aggregated to monthly means. Uncertainty is given by the intra-monthly standard deviation.

Data from the Soil Moisture Active Passive (SMAP) project is used as dynamic prior (Reichle et al. 2014). Specifically, the model-derived value-added Level 4 data product with 3-hourly estimates of soil moisture and respective error estimates at a 9 km resolution are averaged to daily values as the MULTIPLY platform assimilates data at this temporal resolution.




Mattia, F. et al. (2006) Using a priori information to improve soil moisture retrieval from ENVISAT ASAR AP data in semiarid regions. IEEE Trans. Geosci. Remote Sens. 44: 900–912.

Dorigo, W. A., et al., 2017, ESA CCI Soil Moisture for improved Earth system understanding: State-of-the art and future directions, Remote Sensing of Environment, 203, 185-215, 2017, doi:10.1016/j.rse.2017.07.001.

Gruber, A., et al., 2017, Triple Collocation-Based Merging of Satellite Soil Moisture Retrievals, Transactions on Geoscience and Remote Sensing, 55(12), 1-13. doi:10.1109/TGRS.2017.2734070.

Liu, Y. Y., et al., 2012, Trend-preserving blending of passive and active microwave soil moisture retrievals, Remote Sensing of Environment, 123, 280-297.

Reichle, R. et al. 2014. SMAP Algorithm Theoretical Basis Document: L4 Surface and Root-Zone Soil Moisture Product. SMAP Project, JPL D-66483, Jet Propulsion Laboratory, Pasadena, CA, USA.
