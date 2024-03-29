# Breidablik
An interpolation routine and abundance predictor wrapper around stellar spectra for lithium generated from Balder (Amarsi et al. 2016) and the Stagger-grid (Magic et al. 2013). The raw synthetic spectra and models can be found at https://zenodo.org/records/10428805. We use radial basis functions (Bertran de Lis et al. 2022) to interpolate between stellar parameters and lithium abundance inputs to generate interpolated 3D NLTE stellar line profiles; and `pytorch` to interpolate between stellar parameters and stellar line strengths, these interpolation models are provided as part of the package. Using the interpolation routine, we can predict the lithium profile given any stellar parameters and lithium abundance input, we can also predict the lithium abundance given an observed spectrum and stellar parameters.

## Installation
### Install with pip
#### Automatic
If you are using linux/macOS, you can install using:
```
wget https://raw.githubusercontent.com/ellawang44/Breidablik/master/install_pip
./install_pip
```
Once the installation is done, you can delete the installation script.

#### Manual
If the above automatic installation did not work for you, then to install with `pip`, there are 3 steps:
1. Install Breidablik through `pip`. This will install Breidablik _without_ the raw synthetic spectra.  
2. Optional, download the raw data. Whilst Breidablik will happily interpolate and predict lithium abundances without the raw synthetic spectra, the raw spectra can be found in `balder.zip` on https://zenodo.org/records/10428805. There are functions under `breidablik.analysis` which interact with the raw synthetic spectra.
3. Optional, put the raw data in the breidablik folder. The functions in `breidablik.analysis.read` have a `data_path` parameter which is the path to the folder containing the raw spectra. By default, this path is set to a folder named `Balder` inside the `breidablik` package. Therefore, I recommend putting the data inside the `Balder` folder inside the `breidablik` package; however, this is not a requirement.  

### Install without pip
#### Automatic
If you are using linux/macOS, you can navigate to where you want this repository and run:
```
git clone https://github.com/ellawang44/Breidablik
cd Breidablik
./install
```

#### Manual
If the above automatic installation did not work for you, then to install without using `pip`, there are 5 steps:
1. Navigate to where you want this repository and clone this git repository.
2. Navigate into `Breidablik/breidablik`.
3. Download the models from `models_vX.X.X.zip` on https://zenodo.org/records/10428805 and unzip it. Pick the latest model, i.e. the model with the highest version number. If you are running an older version of Breidablik, pick the model based on the version of Breidablik you want to run. The model version number will be just lower than your Breidablik version number. e.g. running Breidablik v1.1.0 will require models v1.0.0. 
4. Optional, download the raw data from `balder.zip` on https://zenodo.org/records/10428805 and unzip it.
5. Optional, add this directory to the python path. This makes the directory findable by python no matter where it is launched.

### Check that your installation is correct
Optional, to check that the installation was successful, in the `Breidablik` folder, you can run:
```
python -m pytest
```
If all tests pass with no warnings, then the installation was successful. If you have installed this package through `pip`, you can still run the tests, but instead this will need to be done in the `breidablik` folder installed by `pip`.

## Updating the code
The easiest way to update the code is by doing a clean install. 

There are 3 components to Breidablik: synthetic spectra from Balder, trained models for interpolation, the code itself. 

The synthetic spectra from Balder is downloaded once upon install, if there were updates to this, reinstall is recommended. Note that old versions of Breidablik installed spectra as separate files (~2000 files), new versions of Breidablik have one numpy file per grid. This change was made because pip takes a while to delete a lot of small files. If you are uninstalling an old version of Breidablik, be aware that it might take a while. 

The trained models for interpolation is downloaded upon manual install, but comes with the package if installed through pip. If you've installed the code manually, you will need to update the trained models manually.

The code itself can be updated both through pip and git if you installed manually. 

## Getting Started
See the examples at https://breidablik.readthedocs.io/en/latest/Getting%20Started.html#examples

## License and Citation
If you use this package for any academic purpose, please cite Wang et al. 2020 (arXiv:2010.15248) and Wang et al. 2024 (arXiv:2402.02669).
