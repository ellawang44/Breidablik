# Breidablik
An interpolation routine and abundance predictor wrapper around stellar spectra for lithium generated from Balder (Amarsi et al. 2016) and the Stagger-grid (Magic et al. 2013). The raw synthetic spectra are not part of the package, however, can be found in `balder.zip` on http://www.mso.anu.edu.au/~ellawang/. We interpolate between stellar parameters and lithium abundance inputs to generate interpolated synthetic spectra, these interpolation models are provided as part of the package. Using the interpolation routine, we can predict the lithium profile given any stellar parameters and lithium abundance input, we can also predict the lithium abundance given an observed spectrum and stellar parameters.

## Installation
### Automatic Installation
To install the interpolation routine and models run:
```
pip3 install breidablik
```
This will install Breidablik _without_ the raw synthetic spectra.

### Data Download
Whilst Breidablik will happily interpolate and predict lithium abundances without the raw synthetic spectra, the raw spectra can be found here: in `balder.zip` on http://www.mso.anu.edu.au/~ellawang/. There are functions under `breidablik.analysis.read` which will read in the raw synthetic spectra, and thus depends on this data.

The functions in `breidablik.analysis.read` have a `data_path` parameter which is the path to the folder containing the raw spectra. By default, this path is set to a folder named `Balder` inside the `breidablik` package. Therefore, I recommend putting the data inside the `Balder` folder inside the `breidablik` package; however, this is not a requirement.  

To put the data inside `breidablik`, we need to locate where `pip3` has installed `breidablik`, this can be done using:
```
pip3 show breidablik
```
The path to `breidablik` should be displayed, navigate to it. Inside this folder, there should already be a folder named `Balder` with `wavelengths.dat` in it. Merge the `Balder` folder containing the other data files with this pre-existing folder.

If the raw data is placed in a folder elsewhere, every function that has optional parameter `data_path` needs to point to this folder location.

### Manual Installation
To manually install Breidablik, there are 4 steps:

1. clone this git repository using:
```
git clone https://github.com/ellawang44/Breidablik
```
2. Navigate into `Breidablik/breidablik`
3. Download the models from `models.zip` on http://www.mso.anu.edu.au/~ellawang/ and create a folder named `models` to place them in.
4. Download the raw data from `balder.zip` on http://www.mso.anu.edu.au/~ellawang/ and _merge_ it into the existing `Balder` folder.

## Getting Started
To be written soon. 
