<img alt="MULTIPLY" align="right" src="https://raw.githubusercontent.com/multiply-org/multiply-core/master/doc/source/_static/logo/Multiply_multicolour.png" />

# MULTIPLY Prior Engine

[![Build Status](https://travis-ci.org/multiply-org/prior-engine.svg?branch=master)](https://travis-ci.org/multiply-org/prior-engine)
[![Documentation Status](https://readthedocs.org/projects/multiply/badge/?version=latest)](http://multiply.readthedocs.io/en/latest/?badge=latest)

<!-- [![Documentation Status](https://readthedocs.org/projects/prior-engine/badge/?version=latest)](http://prior-engine.readthedocs.io/en/latest/?badge=latest) -->

This repository contains the prior engine for the MULTIPLY main platform.
It provides *a priori* information to the [Inference Engine](https://github.com/multiply-org/KaFKA-InferenceEngine) to support land surface parameter retrieval.
The documentation is part of the MULTIPLY core documentation on [ReadTheDocs](http://multiply.readthedocs.io/en/latest/).
<!-- Add plans and current status? -->

## Contents

* `aux_data/` Auxiliary data for prior generation.
* `doc` - The auto generated documentation of all prior engine classes and function definitions.
* `multiply_prior-engine/` - The main prior engine software package.
as source of information and orientation.
* `recipe` Conda installation recipe.
* `test/` - The test package.
* `setup.py` - main build script, to be run with Python 3.6
* `LICENSE.md` - License of software in repository.
<!-- * `helpers/` - Helper functions. -->

## How to install

The first step is to clone the latest code and step into the check out directory:

    $ git clone https://github.com/multiply-org/prior-engine.git
    $ cd prior-engine

The MULTIPLY platform has been developed against Python 3.6.
It cannot be guaranteed to work with previous Python versions.

The MULTIPLY prior engine can be run from sources directly.
To install the MULTIPLY prior engine into an existing Python environment just for the current user, use

    $ python setup.py install --user

To install the MULTIPLY Core for development and for the current user, use

    $ python setup.py develop --user

## Module requirements

- `python-dateutil`
- `gdal`
- `matplotlib`
- `numpy`
- `pyyaml`
- `shapely`


## Usage

### Python Package

MULTIPLY prior engine is available as Python Package.
To import it into your python application, use

```python
import multiply_prior_engine
```

### User defined priors

Users are provided the possibility to choose between prior-types, using the configuration file. This configuration file can be modified by both the users directly (using simple text editors), as well as the user-interface described below and in the upcoming MULTIPLY platform user-interface.

The user has three options to add prior data to the retrieval (in addition to choosing priors already made available by MULTIPLY).

- The user can choose to define single values for the prior in terms of transformed ‘mu’ and ‘unc’ values.
- The user can choose to provide a single geolocated tiff file, with both mean and uncertainty values. Here, the mean value should be provided as the first band, while the uncertainty of these values should be provided as the second band.
- Finally, the user can choose to provide a directory with multiple files, following a similar structure as the previous choice. Here, the files should be given a 8 digit date stamp in the filename.

The configuration file then could look like:

``` yaml
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
```


### Command Line Interface

There is a Command Line Interface (CLI) integrated to allow for the following actions:

- add user defined prior data,
- import user defined prior data,
- remove/un-select prior data from configuration,
- show configuration.

The CLI's help can be accessed via `-h` flag:

``` bash
user_prior -h
```

and will show:

``` bash
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

```

The help and description of the above mentioned sub-commands can be accessed via, e.g.:

``` bash
user_prior add -h
```


---

**NOTE:**

If installed for the current user only, make sure the directory the prior engine gets installed to is in your PATH variable.

---



### Logging

For now the Prior Engine has its own logging setup. To set the `logging level` please adjust the level accordingly in the `multiply_prior_engine/__init__.py` file. Available options are: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL.

## Generating the Documentation

We use [Sphinx](http://www.sphinx-doc.org/en/stable/rest.html) to generate the documentation of the MULTIPLY platform on [ReadTheDocs](http://multiply.readthedocs.io/en/latest/).

The source files of the main documentation of the MULTIPLY platform is to be found in the [MULTIPLY core repository](https://github.com/multiply-org/multiply-core).

If there is a need to build the *prior engine specific* docs locally, these additional software packages are required:

    $ conda install sphinx sphinx_rtd_theme mock
    $ conda install -c conda-forge sphinx-argparse
    $ pip install sphinx_autodoc_annotation

To regenerate the HTML docs, type

    $ cd doc
    $ make html


## Contribution and Development

Once, the package is set up, you are very welcome to contribute to the MULTIPLY Prior Engine.
Please find corresponding guidelines and further information on how to do so in the [CONTRIBUTION.md](https://github.com/multiply-org/prior-engine/blob/master/CONTRIBUTION.md) document.

### Reporting issues and feedback

If you encounter any bugs with the tool, please file a [new issue](https://github.com/multiply-org/prior-engine/issues/new) while adhering to the above mentioned guidelines.



## Authors

* **Joris Timmermans** - *Work on vegetation priors*
* **Thomas Ramsauer** - *Work on soil priors*

<!-- See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project. -->

## License

This project is licensed under the GPLv3 License - see the [LICENSE.md](https://github.com/multiply-org/prior-engine/blob/master/LICENSE.md) file for details.

<!-- ## Acknowledgments -->

<!-- * Alexander Löw for.. -->

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/multiply-org/prior-engine/tags).
