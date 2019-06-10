# Breidablik
An interpolation routine and abundance predictor wrapper around stellar spectra for lithium generated from Balder (Amarsi et al. 2016) and the Stagger-grid (Magic et al. 2013). The raw synthetic spectra and models can be found on http://www.mso.anu.edu.au/~ellawang/. We interpolate between stellar parameters and lithium abundance inputs to generate interpolated synthetic spectra, these interpolation models are provided as part of the package. Using the interpolation routine, we can predict the lithium profile given any stellar parameters and lithium abundance input, we can also predict the lithium abundance given an observed spectrum and stellar parameters.

## Installation
### Install with pip (automatic)
To install the interpolation routine and models with `pip`, there are 3 steps:  
1. Install Breidablik through `pip`. This will install Breidablik _without_ the raw synthetic spectra.  
2. Optional, download the raw data. Whilst Breidablik will happily interpolate and predict lithium abundances without the raw synthetic spectra, the raw spectra can be found in `balder.zip` on http://www.mso.anu.edu.au/~ellawang/. There are functions under `breidablik.analysis.read` which will read in the raw synthetic spectra, and thus depends on this data.  
3. Optional, put the raw data in the breidablik folder. The functions in `breidablik.analysis.read` have a `data_path` parameter which is the path to the folder containing the raw spectra. By default, this path is set to a folder named `Balder` inside the `breidablik` package. Therefore, I recommend putting the data inside the `Balder` folder inside the `breidablik` package; however, this is not a requirement.  

If you are using linux/macOS, then you can install using:
```
git clone https://github.com/ellawang44/Breidablik
cd Breidablik
./install_auto
```

### Install without pip (manual)
To install Breidablik without using `pip`, there are 5 steps:

1. Navigate to where you want this repository and clone this git repository.
2. Navigate into `Breidablik/breidablik`.
3. Download the models from `models.zip` on http://www.mso.anu.edu.au/~ellawang/ and unzip it.
4. Download the raw data from `balder.zip` on http://www.mso.anu.edu.au/~ellawang/ and unzip it.
5. Optional: Add this directory to the python path. This makes the directory findable by python no matter where it is launched.

If you are using linux/macOS, then you can navigate to where you want this repository and run:
```
git clone https://github.com/ellawang44/Breidablik
cd Breidablik
./install_man
```

## Getting Started
To be written soon.
