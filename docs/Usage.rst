Usage
=====

Python Package
------------------

MULTIPLY prior engine is available as Python Package.
To import it into your python application, use

.. code-block:: python

   import multiply_prior_engine

User defined priors
---------------------

Users are provided the possibility to choose between prior-types, using the configuration file. This configuration file can be modified by both the users directly (using simple text editors), as well as the user-interface described below and in the upcoming MULTIPLY platform user-interface.

The user has three options to add prior data to the retrieval (in addition to choosing priors already made available by MULTIPLY).

- The user can choose to define single values for the prior in terms of transformed ‘mu’ and ‘unc’ values.
- The user can choose to provide a single geolocated tiff file, with both mean and uncertainty values. Here, the mean value should be provided as the first band, while the uncertainty of these values should be provided as the second band.
- Finally, the user can choose to provide a directory with multiple files, following a similar structure as the previous choice. Here, the files should be given a 8 digit date stamp in the filename.

The configuration file then could look like:

.. code-block:: yaml

    Prior
	    General:
		    directory_data: ‘path 2 prior engine’
	    LAI:
		    database
			    static_dir: same as general directory_data
	    SM:
		    user:
			    mu: 0.5
			    unc: 0.02
	    CWC:
		    user:
			    file: ‘path to geotiff-file’
	    ALA:
		    user:
			    dir: ‘path to directory with geotiff-files (sorted on date)’

		    ...


	    output_directory: ‘path to outputdirectory’



Command Line Interface
------------------------

There is a Command Line Interface (CLI) integrated to allow for the following actions:

- add user defined prior data,
- import user defined prior data,
- remove/un-select prior data from configuration,
- show configuration.

The CLI's help can be accessed via `-h` flag:

.. code-block:: bash

   user_prior -h

and will show:

.. code-block:: bash

    usage: user_prior.py [-h] {show,S,add,A,remove,R,import,I} ...

    Utility to integrate User Prior data in MULTIPLY Prior Engine

    positional arguments:
    {show,S,add,A,remove,R,import,I}
	show (S)            Show current prior config.
	add (A)             Add prior directory to configuration.
	remove (R)          Remove prior information from configuration.
	import (I)          Import user prior data.

    optional arguments:
    -h, --help            show this help message and exit


The help and description of the above mentioned sub-commands can be accessed via, e.g.:

.. code-block:: bash

   user_prior add -h

.. note::

   If installed for the current user only, make sure the directory the prior engine gets installed to is in your PATH variable.



Logging
---------

For now the Prior Engine has its own logging setup. To set the `logging level` please adjust the level accordingly in the `multiply_prior_engine/__init__.py` file. Available options are: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL.
