# Battery Signal Selection and Enhancement Toolbox

BaSSET is a Python GUI software designed to simplify multivariate analysis of _operando_ scattering experiments through integrating common algorithms with easy access to change parameters and visualize results.

# Installation

This software is available on the [Python Package Index](https://pypi.org/project/BaSSET-UiO/). The simplest way to install is through running `pip install basset-uio`. This also adds two executables for you to run from the terminal or your OS' search. Note that the python 'Scripts' folder needs to be in your PATH for these to work.  
`basset` launches the GUI with a terminal window (or in your active terminal) with print statements and potential warnings or errors.  
`basset-gui` launches the GUI without a terminal (or without occupying your active termainal).  
Other than this, they both act the same.

# Acknowledgements

This package is simply a GUI interface for multivariate analysis methods, with built-in results viewing and export functionality. All credit goes to the creators of the utililzed algorithms.

PCA, NMF and ICA  come from `scikit-learn` [1].  
SNMF comes from `diffpy.stretched-nmf` [2].

[1] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, Édouard Duchesnay, [Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), *Journal of Machine Learning Research* **12**, pp. 2825-2830 (2011).  
[2] Ran Gu, Yevgeny Rakita, Ling Lan, Zach Thatcher, Gabrielle E. Kamm, Daniel O'Nolan, Brennan McBride, Allison Wustrow, James R. Neilson, Karena W. Chapman, Qiang Du, and Simon J. L. Billinge, [Stretched Non-negative Matrix Factorization](https://doi.org/10.1038/s41524-024-01377-5), npj *Comput Mater* **10**, 193 (2024).