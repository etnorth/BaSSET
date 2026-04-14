import sys
import os
from glob import glob
from datetime import datetime
from re import search

from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, NMF, FastICA
from diffpy.stretched_nmf.snmf_class import SNMFOptimizer 
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qtw
import PyQt6.QtGui as qtg


import platform
if platform.system() == "Windows":
    # Separates BaSSET from the "Pythonw.exe" ID so it can have its own tackbar icon
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u"UiO.BaSSET")

uio_palette = ['#86A4F7','#2EC483','#FB6666']
uio_cmp = LinearSegmentedColormap.from_list('UiO colourmap',uio_palette)

def import_data(filename):
    """
    Takes in a filename (str) to find continous two-column datasets, returning each column as a numpy array
    """
    try:
        x, y = np.loadtxt(filename, unpack=True, comments='#', usecols=(0,1))
        return x, y
    except ValueError:
        print("Couldn't read file with deafult # comments, reading file line by line")
        pass
    with open(filename, "r") as infile:
        lines = infile.readlines()
        for i, line in enumerate(lines):
            if not line:
                continue
            if search("[0-9]", line.replace(' ','')[0]): # Checks if the first non-whitespace symbol is a number
                x, y = np.loadtxt(filename, unpack=True, skiprows=i, usecols=(0,1)) # Skips all lines above curren
                return x, y
        raise ValueError(f"Couldn't find data in {filename}")

def import_dataset(dir):
    """
    Imports data from all files in chosen directory of chosen / most popular filetype
    """

    filetypes = {}
    for file_extension in [os.path.splitext(filename)[-1] for filename in glob(f"{dir}*")]: # Gets extension of each filename in folder
        if file_extension in filetypes:
            filetypes[file_extension] += 1
        else:
            filetypes[file_extension] = 1
    filetype = max(filetypes)
    popular_filetypes = [key for key, value in filetypes.items() if value == filetypes[filetype]] # Checks if there are multiple equally popular file extensions
    if len(popular_filetypes)>1:
        print(f"Multiple equally popular filetypes found: ({popular_filetypes}). Disable 'auto' and re-run")
        raise NotImplementedError
    print(f"Most common filetype in supplied directory is {filetype}")

    print(f"Looking for files in \"...{dir[-41:-1]}\" of type {filetype}")
    filenames = natsorted(glob(f"{dir}*{filetype}"))
    print(f"Found {len(filenames)} files of filetype {filetype}")

    x_data = []
    y_data = []

    for filename in filenames:
        x, y = import_data(filename)
        x_data.append(x)
        y_data.append(y)

    print("Dataset imported")
    return np.array(x_data), np.array(y_data)

def theta_to_Q(angles, wavelength):
    return ((4*np.pi) / wavelength) * np.sin(np.deg2rad(angles)/2)

def normalize_dataset(intensities):
    for i in range(len(intensities)):
        intensities[i] = intensities[i] / intensities[i].max()
    return intensities

def PCA_analysis(intensities, numComponents, whiten, svd_solver, tol, iterated_power, n_oversamples, power_iteration_normalizer):
    n_components = min(10,min(np.shape(intensities))) # Setting this higher than the user's number ensures reporting of explained variances
    pca = PCA(n_components = n_components, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, n_oversamples=n_oversamples, power_iteration_normalizer=power_iteration_normalizer)
    X = pca.fit(intensities)
    transformed = pca.transform(intensities)
    reconstructed = pca.inverse_transform(transformed)

    return X, transformed, reconstructed

def NMF_analysis(intensities, numComponents, init, solver, beta_loss, tol, max_iter, alpha_W, alpha_H, l1_ratio):
    n_components_list = np.arange(1, min(10,min(np.shape(intensities)))+1, dtype=int)
    errors = np.empty(len(n_components_list))
    X = None
    transformed = None
    reconstructed = None

    if intensities.min() < 0:
        intensities -= intensities.min()
        print("Negative value found in dataset. Lifting data above zero")

    for i, n_components in enumerate(n_components_list):
        print(f"Calculating NMF reconstruction error for {i} components")
        nmf = NMF(n_components=n_components, init=init, solver=solver, beta_loss=beta_loss, tol=tol, max_iter=max_iter, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio)
        X_temp = nmf.fit(intensities)
        errors[i] = X_temp.reconstruction_err_
        if numComponents == n_components:
            X = X_temp
            transformed = nmf.transform(intensities)
            reconstructed = nmf.inverse_transform(transformed)

    return X, transformed, reconstructed, errors

def ICA_analysis(intensities, numComponents, algorithm, whiten, fun, max_iter, tol, whiten_solver):
    ica = FastICA(n_components = numComponents, algorithm=algorithm, whiten=whiten, fun=fun, max_iter=max_iter, tol=tol, whiten_solver=whiten_solver)
    X = ica.fit(intensities)
    transformed = ica.transform(intensities)
    reconstructed = ica.inverse_transform(transformed)

    return X, transformed, reconstructed

def SNMF_analysis(intensities, numComponents, min_iter, max_iter, tol, rho, eta):
    n_components_list = np.arange(1, min(10,min(np.shape(intensities)))+1, dtype=int)
    errors = np.empty(len(n_components_list))
    X = None
    transformed = None
    reconstructed = None

    # SNMF automatically handles lifts negative values, so no if-case needed

    for i, n_components in enumerate(n_components_list):
        print(f"Calculating SNMF reconstruction error for {n_components} components")
        snmf = SNMFOptimizer(n_components=4, min_iter=min_iter, max_iter=max_iter, tol=tol, rho=rho, eta=eta, show_plots=True)
        print(f"n_components={n_components}, min_iter={min_iter}, max_iter={max_iter}, tol={tol}, rho={rho}, eta={eta}")
        X_temp = snmf.fit(intensities)
        errors[i] = X_temp.reconstruction_err_
        if numComponents == n_components:
            X = X_temp
            transformed = snmf.components_
            scores = snmf.weights_
            stretch = snmf.stretch_
            reconstructed = snmf.reconstruct_matrix(transformed, scores, stretch)

    return X, transformed, reconstructed, scores, stretch

class SciSpinBox(qtw.QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimum(-np.inf)
        self.setMaximum(np.inf)
        self.setDecimals(1000)
        self.validator = qtg.QDoubleValidator()
        self.validator.setNotation(qtg.QDoubleValidator.Notation.ScientificNotation)
        self.lineEdit().setValidator(self.validator)

        def textFromValue(self, value):
            return "{.2e}".format(value)
        
        def valueFromText(self, text):
            try:
                return float(text)
            except ValueError:
                print("Could not convert scientific number user input to float. Deau")
                return 0.0

class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.configpath = os.path.dirname(os.path.realpath(__file__))
        self.configfile = f"{self.configpath}/config.dat"

        self.setWindowIcon(qtg.QIcon(f"{self.configpath}/assets/icon.ico"))
        self.setWindowTitle("Battery Signal Selection and Enhancement Toolbox")
        self.resize(qtc.QSize(640,480)) # Increase to 640x480 in future?
        self.central_widget = qtw.QWidget(self)
        self.grid = qtw.QGridLayout()
        self.central_widget.setLayout(self.grid)
        self.setCentralWidget(self.central_widget)

        ##############################
        ##### Input data widgets #####
        ##############################
        self.indir_layout = qtw.QHBoxLayout()
        self.grid.addLayout(self.indir_layout, 0,0,1,3)

        self.indirLabel = qtw.QLabel("Select the folder containing your data files")
        self.indir_layout.addWidget(self.indirLabel)
        self.indirLabel.setAlignment(qtc.Qt.AlignmentFlag.AlignRight)
        self.indirLabel.setMinimumWidth(350)
        self.indirLabel.setSizePolicy(qtw.QSizePolicy.Policy.Preferred, qtw.QSizePolicy.Policy.Fixed)
        self.indirLabel.setFrameShape(qtw.QFrame.Shape.Panel)
        self.indirLabel.setFrameShadow(qtw.QFrame.Shadow.Sunken)

        self.indirButton = qtw.QPushButton("File directory")
        self.indirButton.setSizePolicy(qtw.QSizePolicy.Policy.Fixed, qtw.QSizePolicy.Policy.Fixed)
        self.indir_layout.addWidget(self.indirButton)

        self.input_format_layout = qtw.QGridLayout()
        self.grid.addLayout(self.input_format_layout, 1,0)

        self.inputformatTitle = qtw.QLabel("Input format")
        self.inputformatTitle.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.inputformatTitle.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.input_format_layout.addWidget(self.inputformatTitle, 0,0,1,3)

        self.inputformatGroup = qtw.QButtonGroup()
        self.thetaButton = qtw.QRadioButton("2θ (°)")
        self.thetaButton.setToolTip("Uses supplied wavelength to display data in Q (Å⁻¹)")
        self.inputformatGroup.addButton(self.thetaButton)
        self.input_format_layout.addWidget(self.thetaButton, 1,0)

        self.qButton = qtw.QRadioButton("Q (Å⁻¹)")
        self.qButton.setChecked(True)
        self.inputformatGroup.addButton(self.qButton)
        self.input_format_layout.addWidget(self.qButton, 1,1)

        self.rButton = qtw.QRadioButton("r (Å)")
        self.inputformatGroup.addButton(self.rButton)
        self.input_format_layout.addWidget(self.rButton, 1,2)

        self.indir_options_layout = qtw.QHBoxLayout()
        self.input_format_layout.addLayout(self.indir_options_layout, 2,0,1,3)

        self.convert2QCheckbox = qtw.QCheckBox("Convert to Q")
        self.indir_options_layout.addWidget(self.convert2QCheckbox)
        self.inputformatGroup.buttonClicked.connect(
            lambda button: self.convert2QCheckbox.setEnabled(True)
            if button==self.thetaButton
            else self.convert2QCheckbox.setDisabled(True)
        )

        self.wavelengthWidget = qtw.QDoubleSpinBox(decimals=6)
        self.wavelengthWidget.setMinimum(0)
        self.wavelengthWidget.setSingleStep(0.000001)
        self.wavelengthWidget.setSuffix(' Å')
        self.indir_options_layout.addWidget(self.wavelengthWidget)
        self.inputformatGroup.buttonClicked.connect(
            lambda button: self.wavelengthWidget.setEnabled(True)
            if button==self.thetaButton
            else self.wavelengthWidget.setDisabled(True)
        )

        #############################
        ##### Algorithm widgets #####
        #############################

        self.algorithm_layout = qtw.QGridLayout()
        self.grid.addLayout(self.algorithm_layout, 1,1)

        self.algorithmGroup = qtw.QButtonGroup()
        self.algorithmTitle = qtw.QLabel("Algorithm")
        self.algorithmTitle.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.algorithmTitle.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.algorithm_layout.addWidget(self.algorithmTitle, 0,0,1,3)

        self.PCAButton = qtw.QRadioButton("PCA")
        self.PCAButton.setChecked(True)
        self.PCAButton.setToolTip("Principal Component Analysis (scikit-learn)")
        self.algorithm_layout.addWidget(self.PCAButton, 1,0)

        self.NMFButton = qtw.QRadioButton("NMF")
        self.NMFButton.setToolTip("Non-Negative Matrix Factorization (scikit-learn)")
        self.algorithm_layout.addWidget(self.NMFButton, 1,1)

        self.ICAButton = qtw.QRadioButton("ICA")
        self.ICAButton.setToolTip("Independent Component Analysis (scikit-learn)")
        self.algorithm_layout.addWidget(self.ICAButton, 1,2)

        self.SNMFButton = qtw.QRadioButton("SNMF")
        self.SNMFButton.setToolTip("Stretched Non-Negative Matrix Factorization (diffpy)")
        self.algorithm_layout.addWidget(self.SNMFButton, 2,0)

        self.algorithmGroup.addButton(self.PCAButton)
        self.algorithmGroup.addButton(self.NMFButton)
        self.algorithmGroup.addButton(self.ICAButton)
        self.algorithmGroup.addButton(self.SNMFButton)

        self.num_components_layout = qtw.QGridLayout()
        self.grid.addLayout(self.num_components_layout, 2,1)

        self.numComponentsTitle = qtw.QLabel("Number of Components")
        self.numComponentsTitle.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.numComponentsTitle.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.num_components_layout.addWidget(self.numComponentsTitle, 0,0)

        self.num_components_slider_layout = qtw.QHBoxLayout()
        self.num_components_layout.addLayout(self.num_components_slider_layout, 1,0)

        self.numComponentsSlider = qtw.QSlider(qtc.Qt.Orientation.Horizontal)
        self.numComponentsSlider.setTickPosition(qtw.QSlider.TickPosition.TicksBothSides)
        self.numComponentsSlider.setTickInterval(1)
        self.numComponentsSlider.setMinimum(2)
        self.numComponentsSlider.setMaximum(10)
        self.numComponentsSlider.setValue(4)
        self.numComponentsSlider.setSingleStep(1)
        self.numComponentsSlider.setPageStep(1)
        self.numComponentsSlider.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.num_components_slider_layout.addWidget(self.numComponentsSlider)

        self.numComponentsLabel = qtw.QLabel(str(self.numComponentsSlider.value()))
        self.numComponentsLabel.setSizePolicy(qtw.QSizePolicy.Policy.Fixed, qtw.QSizePolicy.Policy.Fixed)
        self.num_components_slider_layout.addWidget(self.numComponentsLabel)


        self.algorithm_parameters_layout = qtw.QGridLayout()
        self.grid.addLayout(self.algorithm_parameters_layout, 3,1)

        self.algorithmParametersTitle = qtw.QLabel("Algorithm Parameters")
        self.algorithmParametersTitle.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.algorithmParametersTitle.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.algorithm_parameters_layout.addWidget(self.algorithmParametersTitle, 0,0, 1,3)

        ##########################
        ##### PCA parameters #####
        ##########################
        self.PCAwhitenCheck = qtw.QCheckBox("Whiten")
        self.algorithm_parameters_layout.addWidget(self.PCAwhitenCheck, 1,0)
        self.PCAwhitenCheck.setToolTip("[whiten]\n" \
        "Whitening will remove some information from the transformed signal\n" \
        "(the relative variance scales of the components) but can sometime\n" \
        "improve the predictive accuracy of the downstream estimators\n" \
        "by making their data respect some hard-wired assumptions")

        self.PCAsolverDropdown = qtw.QComboBox()
        self.PCAsolverDropdown.addItems(['auto', 'full', 'covariance_eigh', 'arpack', 'randomized'])
        self.algorithm_parameters_layout.addWidget(self.PCAsolverDropdown, 1,1)
        self.PCAsolverDropdown.setToolTip("[svd_solver]\n" \
        "The type of Singular Value Decomposition solver to use:\n" \
        "auto (default): Chooses solver based on size of dataset and number of components\n" \
        "full: Runs exact full SVD\n" \
        "covariance_eigh: Precomputes covarience for eigenvalue decompositon.\n" \
        "    Efficient for many scans of few datapoints (rare for scattering)\n" \
        "arpack: Runs SVD truncated to number of components.\n" \
        "    Requires fewer components than number of scans\n" \
        "randomized: Runs randomized SVD")

        # Only for 'arpack' solver
        self.PCAtolSpinbox = qtw.QSpinBox()
        self.PCAtolSpinbox.setPrefix("1e")
        self.PCAtolSpinbox.setMinimum(-20)
        self.PCAtolSpinbox.setMaximum(0)
        self.PCAtolSpinbox.setValue(0)
        self.PCAtolSpinbox.setSingleStep(1)
        self.algorithm_parameters_layout.addWidget(self.PCAtolSpinbox, 1,2)
        self.PCAtolSpinbox.setToolTip("[tol]\n" \
        "Tolerance for singular values using 'arpack'")
        self.PCAsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.PCAtolSpinbox.show()
            if currentText=='arpack'
            else self.PCAtolSpinbox.hide()
        )

        # Only for 'randomized' solver
        self.PCAiterated_powerSpinbox = qtw.QSpinBox()
        self.PCAiterated_powerSpinbox.setMinimum(0)
        self.PCAiterated_powerSpinbox.setMaximum(999999)
        self.PCAiterated_powerSpinbox.setValue(0)
        self.PCAiterated_powerSpinbox.setSingleStep(100)
        self.algorithm_parameters_layout.addWidget(self.PCAiterated_powerSpinbox, 2,0)
        self.PCAiterated_powerSpinbox.setToolTip("[iterated_power]\n" \
        "Number of iterations for the power method in 'randomized'")
        self.PCAsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.PCAiterated_powerSpinbox.show()
            if currentText=='randomized'
            else self.PCAiterated_powerSpinbox.hide()
        )

        # Only for 'randomized' solver
        self.PCAiterated_powerAutoCheckbox = qtw.QCheckBox("auto")
        self.algorithm_parameters_layout.addWidget(self.PCAiterated_powerAutoCheckbox, 2,1)
        self.PCAiterated_powerAutoCheckbox.setToolTip("[iterated_power]\n" \
        "Automatically choose number of iterations")
        self.PCAiterated_powerAutoCheckbox.clicked.connect(
            lambda state: self.PCAiterated_powerSpinbox.setDisabled(True)
            if state
            else self.PCAiterated_powerSpinbox.setDisabled(False)
        )
        self.PCAsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.PCAiterated_powerAutoCheckbox.show()
            if currentText=='randomized'
            else self.PCAiterated_powerAutoCheckbox.hide()
        )

        # Only for 'randomized' solver
        self.PCAn_oversampledSpinbox = qtw.QSpinBox()
        self.PCAn_oversampledSpinbox.setMinimum(0)
        self.PCAn_oversampledSpinbox.setMaximum(50)
        self.PCAn_oversampledSpinbox.setValue(10)
        self.PCAn_oversampledSpinbox.setSingleStep(1)
        self.algorithm_parameters_layout.addWidget(self.PCAn_oversampledSpinbox, 2,2)
        self.PCAn_oversampledSpinbox.setToolTip("[n_oversamples]\n" \
        "Additional number of random vectors to sample using 'randomized'")
        self.PCAsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.PCAn_oversampledSpinbox.show()
            if currentText=='randomized'
            else self.PCAn_oversampledSpinbox.hide()
        )

        # Only for 'randomized' solver // Not for 'arpack' solver
        self.PCApower_iteration_normalizerDropdown = qtw.QComboBox()
        self.PCApower_iteration_normalizerDropdown.addItems(['auto', 'QR', 'LU', 'none'])
        self.algorithm_parameters_layout.addWidget(self.PCApower_iteration_normalizerDropdown, 1,2)
        self.PCApower_iteration_normalizerDropdown.setToolTip("[power_iteration_normalizer]\n" \
        "Power iteration normalizer using 'randomized'")
        self.PCAsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.PCApower_iteration_normalizerDropdown.show()
            if currentText=='randomized'
            else self.PCApower_iteration_normalizerDropdown.hide()
        )

        # Ignored PCA parameters:
        # random_state

        self.PCAalgorithmWidgets = [self.PCAwhitenCheck, self.PCAsolverDropdown,
                                    self.PCAtolSpinbox, self.PCAiterated_powerSpinbox,
                                    self.PCAiterated_powerAutoCheckbox, self.PCAn_oversampledSpinbox,
                                    self.PCAn_oversampledSpinbox, self.PCApower_iteration_normalizerDropdown]
        
        ##########################
        ##### NMF parameters #####
        ##########################
        self.NMFinitDropdown = qtw.QComboBox()
        self.NMFinitDropdown.addItems(["nndsvda", "random", "nndsvd", "nndsvdar"])
        self.algorithm_parameters_layout.addWidget(self.NMFinitDropdown, 1,0)
        self.NMFinitDropdown.setToolTip("[init]\n" \
        "Method used to initialize the procedure:\n" \
        "('nndsvda' is reccommended for XRD and 'nndsvd' is recommended for PDF)\n" \
        "nndsvda (default): Better when sparsity is not desired\n" \
        "random: Random non-negative matrices\n" \
        "nndsvd: Non-negative Double Singular Value Decomposition (better for sparseness)\n" \
        "nndsvdar: Faster, less accurate alternative to NNDSVDa when sparsity is not desired")

        self.NMFsolverDropdown = qtw.QComboBox()
        self.NMFsolverDropdown.addItems(["cd", "mu"])
        self.algorithm_parameters_layout.addWidget(self.NMFsolverDropdown, 1,1)
        self.NMFsolverDropdown.setToolTip("[solver]\n" \
        "Numerical solver to use:\n" \
        "cd (default): Coordinate Descent\n" \
        "mu: Multiplicative Update")

        self.NMFmax_iterSpinbox = qtw.QSpinBox()
        self.NMFmax_iterSpinbox.setMinimum(0)
        self.NMFmax_iterSpinbox.setMaximum(99999)
        self.NMFmax_iterSpinbox.setValue(2500)
        self.algorithm_parameters_layout.addWidget(self.NMFmax_iterSpinbox, 2,0)
        self.NMFmax_iterSpinbox.setToolTip("[max_iter]\n" \
        "Maximum number of iterations before timing out")

        self.NMFtolSpinbox = qtw.QSpinBox()
        self.NMFtolSpinbox.setPrefix("1e")
        self.NMFtolSpinbox.setMinimum(-20)
        self.NMFtolSpinbox.setMaximum(0)
        self.NMFtolSpinbox.setValue(-4)
        self.NMFtolSpinbox.setSingleStep(1)
        self.algorithm_parameters_layout.addWidget(self.NMFtolSpinbox, 2,1)
        self.NMFtolSpinbox.setToolTip("[tol]\n" \
        "Tolerance of the stopping condition")

        # Used in 'cd' solver
        self.NMFl1_ratioDSpinbox = qtw.QDoubleSpinBox()
        self.NMFl1_ratioDSpinbox.setMinimum(0)
        self.NMFl1_ratioDSpinbox.setMaximum(1)
        self.NMFl1_ratioDSpinbox.setValue(0)
        self.NMFl1_ratioDSpinbox.setSingleStep(0.05)
        self.algorithm_parameters_layout.addWidget(self.NMFl1_ratioDSpinbox, 1,2)
        self.NMFl1_ratioDSpinbox.setToolTip("[l1_ratio]\n" \
        "Regularization mixing parameter:\n" \
        "(0, default): elementwise l2 penalty aka. Frobenius Norm\n"
        "(1): elementwwise l1 penalty (1)\n")
        self.NMFsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.NMFl1_ratioDSpinbox.show()
            if currentText=='cd'
            else self.NMFl1_ratioDSpinbox.hide()
        )

        # Only for 'mu' solver 
        self.NMFbeta_lossDropdown = qtw.QComboBox()
        self.NMFbeta_lossDropdown.addItems(["frobenius", "kullback-leibler"]) # "itakura-saito" not included as it cannot have zeros in input data, which XRD/PDF often can have
        self.algorithm_parameters_layout.addWidget(self.NMFbeta_lossDropdown, 1,2)
        self.NMFbeta_lossDropdown.setToolTip("[beta_loss]\n" \
        "Beta divergence to be minimized," \
        "measuring the distance between X and the dot product WH using 'mu':\n" \
        "frobenius is default")
        self.NMFsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.NMFbeta_lossDropdown.show()
            if currentText=='mu'
            else self.NMFbeta_lossDropdown.hide()
        )

        self.NMFalpha_WDSpinbox = qtw.QDoubleSpinBox()
        self.NMFalpha_WDSpinbox.setMinimum(0)
        self.NMFalpha_WDSpinbox.setMaximum(10)
        self.NMFalpha_WDSpinbox.setValue(0)
        self.NMFalpha_WDSpinbox.setDecimals(5)
        self.NMFalpha_WDSpinbox.setSingleStep(0.00005)
        self.algorithm_parameters_layout.addWidget(self.NMFalpha_WDSpinbox, 3,0)
        self.NMFalpha_WDSpinbox.setToolTip("[alpha_W]\n" \
        "Constant that multiplies the regularization terms of the features:\n" \
        "(Regularization a penalty term to constrain parameter complexity and reduce overfitting)\n" \
        "0 (default) means no regularization")

        self.NMFalpha_HDSpinbox = qtw.QDoubleSpinBox()
        self.NMFalpha_HDSpinbox.setMinimum(0)
        self.NMFalpha_HDSpinbox.setMaximum(10)
        self.NMFalpha_HDSpinbox.setValue(0)
        self.NMFalpha_HDSpinbox.setDecimals(5)
        self.NMFalpha_HDSpinbox.setSingleStep(0.00005)
        self.algorithm_parameters_layout.addWidget(self.NMFalpha_HDSpinbox, 3,1)
        self.NMFalpha_HDSpinbox.setToolTip("[alpha_H]\n" \
        "Constant that multiplies the regularization terms of the feature mixing:\n" \
        "0 means no regularization")

        self.NMFalpha_HsameCheckbox = qtw.QCheckBox("same")
        self.algorithm_parameters_layout.addWidget(self.NMFalpha_HsameCheckbox, 3,2)
        self.NMFalpha_HsameCheckbox.setToolTip("[alpha_H]\n" \
        "Use same regularization constant for features and mixing")
        self.NMFalpha_HsameCheckbox.clicked.connect(
            lambda state: self.NMFalpha_HDSpinbox.setDisabled(True)
            if state
            else self.NMFalpha_HDSpinbox.setDisabled(False)
        )
        self.NMFalpha_HsameCheckbox.clicked.connect(
            lambda state: self.NMFalpha_HDSpinbox.setValue(self.NMFalpha_WDSpinbox.value())
            if state
            else None
        )
        self.NMFalpha_WDSpinbox.valueChanged.connect(
            lambda value: self.NMFalpha_HDSpinbox.setValue(value)
            if self.NMFalpha_HsameCheckbox.isChecked()
            else None
        )

        # Ignored NMF parmeters:
        # random_state
        # verbose
        # shuffle

        self.NMFalgorithmWidgets = [self.NMFinitDropdown, self.NMFsolverDropdown,
                                    self.NMFmax_iterSpinbox, self.NMFtolSpinbox,
                                    self.NMFl1_ratioDSpinbox, self.NMFbeta_lossDropdown,
                                    self.NMFalpha_WDSpinbox, self.NMFalpha_HDSpinbox,
                                    self.NMFalpha_HsameCheckbox]
        
        ##########################
        ##### ICA parameters #####
        ##########################
        self.ICAalgorithmDropdown = qtw.QComboBox()
        self.ICAalgorithmDropdown.addItems(['parallel', 'deflation'])
        self.algorithm_parameters_layout.addWidget(self.ICAalgorithmDropdown, 1,0)
        self.ICAalgorithmDropdown.setToolTip("[algorithm]\n" \
        "Specify which algorithm to use:\n" \
        "parallel is default")

        self.ICAwhitenDropdown = qtw.QComboBox()
        self.ICAwhitenDropdown.addItems(['unit-variance', 'arbitrary-variance', 'False'])
        self.algorithm_parameters_layout.addWidget(self.ICAwhitenDropdown, 1,1)
        self.ICAwhitenDropdown.setToolTip("[whiten]\n" \
        "Whitening strategy to use. False means no whitening:\n" \
        "unit-variance (default): the whitening matrix is rescaled to ensure each recovered source has unit variance\n" \
        "arbitrary-variance: a whitening with variance arbitrary is used\n" \
        "False: Data is considered whitened and no whitening is performed")

        self.ICAmax_iterSpinbox = qtw.QSpinBox()
        self.ICAmax_iterSpinbox.setMinimum(0)
        self.ICAmax_iterSpinbox.setMaximum(99999)
        self.ICAmax_iterSpinbox.setValue(2500)
        self.algorithm_parameters_layout.addWidget(self.ICAmax_iterSpinbox, 2,0)
        self.ICAmax_iterSpinbox.setToolTip("[max_iter]\n" \
        "Maximum number of iterations during fit")

        self.ICAtolSpinbox = qtw.QSpinBox()
        self.ICAtolSpinbox.setPrefix("1e")
        self.ICAtolSpinbox.setMinimum(-20)
        self.ICAtolSpinbox.setMaximum(0)
        self.ICAtolSpinbox.setValue(-4)
        self.ICAtolSpinbox.setSingleStep(1)
        self.algorithm_parameters_layout.addWidget(self.ICAtolSpinbox, 2,1)
        self.ICAtolSpinbox.setToolTip("[tol]\n" \
        "A positive scalar giving the tolerance at which the un-mixing matrix is considered to have converged")

        self.ICAfunDropdown = qtw.QComboBox()
        self.ICAfunDropdown.addItems(['logcosh', 'exp', 'cube'])
        self.algorithm_parameters_layout.addWidget(self.ICAfunDropdown, 1,2)
        self.ICAfunDropdown.setToolTip("[fun]\n" \
        "The functional form of the G function used in the approximation to neg-entropy:\n" \
        "logcosh is default")

        self.ICAwhiten_solverDropdown = qtw.QComboBox()
        self.ICAwhiten_solverDropdown.addItems(['svd', 'eigh'])
        self.algorithm_parameters_layout.addWidget(self.ICAwhiten_solverDropdown, 2,2)
        self.ICAwhiten_solverDropdown.setToolTip("[whiten_solver]\n" \
        "The solver to use for whitening:\n" \
        "svd (default): More numerically stable if the problem is degenerate, and often faster for scattering\n" \
        "eigh: More memory efficient when there are more scans than datapoints (rare in scattering)")

        # Ignored ICA parameters:
        # fun_args
        # w_init
        # random_state

        self.ICAalgorithmWidgets = [self.ICAalgorithmDropdown, self.ICAwhitenDropdown,
                                    self.ICAmax_iterSpinbox, self.ICAtolSpinbox,
                                    self.ICAfunDropdown, self.ICAwhiten_solverDropdown]
        
        ###########################
        ##### SNMF parameters #####
        ###########################
        self.SNMFmin_iterSpinbox = qtw.QSpinBox()
        self.SNMFmin_iterSpinbox.setMinimum(0)
        self.SNMFmin_iterSpinbox.setMaximum(99999)
        self.SNMFmin_iterSpinbox.setValue(20)
        self.algorithm_parameters_layout.addWidget(self.SNMFmin_iterSpinbox, 2,0)
        self.SNMFmin_iterSpinbox.setToolTip("[min_iter]\n" \
        "Minimum number of iterations before terminating optimzation")

        self.SNMFmax_iterSpinbox = qtw.QSpinBox()
        self.SNMFmax_iterSpinbox.setMinimum(0)
        self.SNMFmax_iterSpinbox.setMaximum(99999)
        self.SNMFmax_iterSpinbox.setValue(500)
        self.algorithm_parameters_layout.addWidget(self.SNMFmax_iterSpinbox, 2,1)
        self.SNMFmax_iterSpinbox.setToolTip("[max_iter]\n" \
        "Maximum number of iterations before terminating optimzation")

        self.SNMFtol_SciSpinbox = SciSpinBox()
        self.SNMFtol_SciSpinbox.setValue(5e-07)
        #self.SNMFtol_SciSpinbox.setPrefix("1e")
        #self.SNMFtol_SciSpinbox.setMinimum(-20)
        #self.SNMFtol_SciSpinbox.setMaximum(0)
        #self.SNMFtol_SciSpinbox.setSingleStep(1)
        self.algorithm_parameters_layout.addWidget(self.SNMFtol_SciSpinbox, 2,2)
        self.SNMFtol_SciSpinbox.setToolTip("[tol] (you can write scientific (5e-07) in this box)\n" \
        "Convergence threshold.\n" \
        "Minimum fractional improvment to allow terminating optimization")

        self.SNMFrho_SciSpinbox = SciSpinBox()
        self.SNMFrho_SciSpinbox.setMinimum(0)
        self.SNMFrho_SciSpinbox.setValue(0)
        self.SNMFrho_SciSpinbox.setDecimals(0)
        #self.SNMFrho_Spinbox.setMaximum(10000000000)
        #self.SNMFrho_Spinbox.setSingleStep(1)
        self.algorithm_parameters_layout.addWidget(self.SNMFrho_SciSpinbox, 3,0)
        self.SNMFrho_SciSpinbox.setToolTip("[rho] (you can write scientific (2e3) in this box)\n"
        "Stretching regularization hyperparameter.\n" \
        "Typically adjusted in powers of 10.\n" \
        "Zero (default) corresponds to no stretching")

        self.SNMFeta_DSpinbox = qtw.QDoubleSpinBox()
        self.SNMFeta_DSpinbox.setMinimum(0)
        self.SNMFeta_DSpinbox.setValue(0)
        self.SNMFeta_DSpinbox.setDecimals(5)
        #self.SNMFeta_DSpinbox.setMaximum(10)
        #self.SNMFeta_DSpinbox.setSingleStep(0.00005)
        self.algorithm_parameters_layout.addWidget(self.SNMFeta_DSpinbox, 3,1)
        self.SNMFeta_DSpinbox.setToolTip("[rho]\n" \
        "Sparsity factor. Should be set to zero (default) for non-sparse data such as PDF.\n" \
        "Can be used to improve results for sparse data such as XRD,\nbut due to instability should be used only faster selecting the best value for rho.\n" \
        "Suggested adjustment is by powers of 2")

        # Ignored SNMF parmeters:
        # random_state
        # verbose
        # show_plots

        self.SNMFwarningLabel = qtw.QLabel("WARNING: The SNMF algorithm takes a lot of time.\n" \
        "It's recommended to decide number of components through other algorithms first.\n" \
        "Error calculation will not be performed.")
        self.SNMFwarningLabel.setWordWrap(True)
        self.grid.addWidget(self.SNMFwarningLabel, 3, 0)
        
        self.SNMFalgorithmWidgets = [self.SNMFmin_iterSpinbox, self.SNMFmax_iterSpinbox,
                                     self.SNMFtol_SciSpinbox, self.SNMFrho_SciSpinbox,
                                     self.SNMFeta_DSpinbox, self.SNMFwarningLabel]
        
        #####################################
        ##### Analysis and plot widgets #####
        #####################################

        self.results_layout = qtw.QGridLayout()
        self.grid.addLayout(self.results_layout, 1,2)

        self.runAnalysisButton = qtw.QPushButton("Run analysis")
        self.runAnalysisButton.setSizePolicy(qtw.QSizePolicy.Policy.MinimumExpanding, qtw.QSizePolicy.Policy.MinimumExpanding)
        self.results_layout.addWidget(self.runAnalysisButton, 0,0,2,2)

        self.export_results_checkbox = qtw.QCheckBox("Export results")
        self.results_layout.addWidget(self.export_results_checkbox, 2,0,1,2)

        self.xrange_layout = qtw.QGridLayout()
        self.grid.addLayout(self.xrange_layout, 2,2)

        self.plot_xrange_label = qtw.QLabel("X-axis range")
        self.plot_xrange_label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.plot_xrange_label.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.xrange_layout.addWidget(self.plot_xrange_label, 2,0,1,2)

        self.plot_xmin_DSpinBox = qtw.QDoubleSpinBox()
        self.plot_xmin_DSpinBox.setMinimum(0)
        self.plot_xmin_DSpinBox.setMaximum(999)
        self.plot_xmin_DSpinBox.setValue(0)
        self.plot_xmin_DSpinBox.setDecimals(2)
        self.plot_xmin_DSpinBox.setSingleStep(0.01)
        self.xrange_layout.addWidget(self.plot_xmin_DSpinBox, 3,0)
        self.plot_xmin_DSpinBox.setToolTip("Minimum x-value in plots")

        self.plot_xmax_DSpinBox = qtw.QDoubleSpinBox()
        self.plot_xmax_DSpinBox.setMinimum(0)
        self.plot_xmax_DSpinBox.setMaximum(999)
        self.plot_xmax_DSpinBox.setValue(30)
        self.plot_xmax_DSpinBox.setDecimals(2)
        self.plot_xmax_DSpinBox.setSingleStep(0.01)
        self.xrange_layout.addWidget(self.plot_xmax_DSpinBox, 3,1)
        self.plot_xmax_DSpinBox.setToolTip("Maximum x-value in plots")
        
        self.plot_layout = qtw.QGridLayout()
        self.grid.addLayout(self.plot_layout, 3,2)

        self.reconstruct_label = qtw.QLabel("Plot Reconstruction Scan")
        self.reconstruct_label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.reconstruct_label.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.plot_layout.addWidget(self.reconstruct_label, 4,0,1,2)

        self.reconstruct_widgets = []

        self.reconstruct_widget0 = qtw.QSpinBox()
        self.reconstruct_widget0.setMinimum(0)
        self.reconstruct_widget0.setMaximum(9999)
        self.plot_layout.addWidget(self.reconstruct_widget0, 5,0)
        self.reconstruct_widget0.setToolTip("If any are '0', will reconstruct uniform distribution of scans")
        self.reconstruct_widgets.append(self.reconstruct_widget0)

        self.reconstruct_widget1 = qtw.QSpinBox()
        self.reconstruct_widget1.setMinimum(0)
        self.reconstruct_widget1.setMaximum(9999)
        self.plot_layout.addWidget(self.reconstruct_widget1, 5,1)
        self.reconstruct_widgets.append(self.reconstruct_widget1)
        
        self.reconstruct_widget2 = qtw.QSpinBox()
        self.reconstruct_widget2.setMinimum(0)
        self.reconstruct_widget2.setMaximum(9999)
        self.plot_layout.addWidget(self.reconstruct_widget2, 6,0)
        self.reconstruct_widgets.append(self.reconstruct_widget2)

        self.reconstruct_widget3 = qtw.QSpinBox()
        self.reconstruct_widget3.setMinimum(0)
        self.reconstruct_widget3.setMaximum(9999)
        self.plot_layout.addWidget(self.reconstruct_widget3, 6,1)
        self.reconstruct_widgets.append(self.reconstruct_widget3)

        self.reconstruct_widget4 = qtw.QSpinBox()
        self.reconstruct_widget4.setMinimum(0)
        self.reconstruct_widget4.setMaximum(9999)
        self.plot_layout.addWidget(self.reconstruct_widget4, 7,0)
        self.reconstruct_widgets.append(self.reconstruct_widget4)

        self.reconstruct_widget5 = qtw.QSpinBox()
        self.reconstruct_widget5.setMinimum(0)
        self.reconstruct_widget5.setMaximum(9999)
        self.plot_layout.addWidget(self.reconstruct_widget5, 7,1)
        self.reconstruct_widgets.append(self.reconstruct_widget5)

        self.reconstruct_widget6 = qtw.QSpinBox()
        self.reconstruct_widget6.setMinimum(0)
        self.reconstruct_widget6.setMaximum(9999)
        self.plot_layout.addWidget(self.reconstruct_widget6, 8,0)
        self.reconstruct_widgets.append(self.reconstruct_widget6)

        self.reconstruct_widget7 = qtw.QSpinBox()
        self.reconstruct_widget7.setMinimum(0)
        self.reconstruct_widget7.setMaximum(9999)
        self.plot_layout.addWidget(self.reconstruct_widget7, 8,1)
        self.reconstruct_widgets.append(self.reconstruct_widget7)

        self.reconstruct_widget8 = qtw.QSpinBox()
        self.reconstruct_widget8.setMinimum(0)
        self.reconstruct_widget8.setMaximum(9999)
        self.plot_layout.addWidget(self.reconstruct_widget8, 9,0)
        self.reconstruct_widgets.append(self.reconstruct_widget8)

        self.reconstruct_widget9 = qtw.QSpinBox()
        self.reconstruct_widget9.setMinimum(0)
        self.reconstruct_widget9.setMaximum(9999)
        self.plot_layout.addWidget(self.reconstruct_widget9, 9,1)
        self.reconstruct_widgets.append(self.reconstruct_widget9)

        ################################
        ##### Function connections #####
        ################################
        self.indirButton.clicked.connect(self.set_indir)
        self.indirButton.clicked.connect(self.update_config_file)
        self.inputformatGroup.buttonClicked.connect(self.update_config_file)
        self.wavelengthWidget.valueChanged.connect(self.update_config_file)
        self.algorithmGroup.buttonClicked.connect(self.display_algorithm_widgets)
        self.algorithmGroup.buttonClicked.connect(self.update_config_file)
        self.numComponentsSlider.valueChanged.connect(lambda value: self.numComponentsLabel.setText(str(value)))
        self.numComponentsSlider.valueChanged.connect(self.update_config_file)
        self.numComponentsSlider.valueChanged.connect(self.display_reconstruction_widgets)
        self.runAnalysisButton.clicked.connect(self.run_analysis)

        ########################
        ##### Top bar menu #####
        ########################
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        new_indir_button = qtg.QAction("Set file directory...", self)
        new_indir_button.triggered.connect(self.set_indir)
        new_indir_button.triggered.connect(self.update_config_file)
        file_menu.addAction(new_indir_button)
        file_menu.addAction(new_indir_button)
        open_indir_button = qtg.QAction("Open file directory", self)
        open_indir_button.triggered.connect(lambda: qtg.QDesktopServices.openUrl(qtc.QUrl.fromLocalFile(self.indirLabel.text())))
        file_menu.addAction(open_indir_button)

        config_button = qtg.QAction("Open config file", self)
        config_button.triggered.connect(lambda: qtg.QDesktopServices.openUrl(qtc.QUrl.fromLocalFile(f"{self.configpath}/config.dat")))
        file_menu.addAction(config_button)
        
        file_menu.addSeparator()
        
        self.file_submenu_recent = file_menu.addMenu("Choose recent")
        
        self.file_submenu_recent_separator = self.file_submenu_recent.addSeparator()

        self.clear_recent_button = qtg.QAction("Clear recently opened", self)
        #self.clear_recent_button.triggered.connect(self.clear_recent_dirs)
        self.clear_recent_button.triggered.connect(lambda: [self.file_submenu_recent.removeAction(action) for action in self.file_submenu_recent.actions()[:-2]])
        self.clear_recent_button.triggered.connect(self.update_config_file)
        self.file_submenu_recent.addAction(self.clear_recent_button)

        file_menu.addSeparator()

        exit_button = qtg.QAction("Exit", self)
        exit_button.triggered.connect(self.close_event)
        file_menu.addAction(exit_button)
        

        help_menu = menu.addMenu("Help")
        github_button = qtg.QAction("Source code", self)
        github_button.triggered.connect(lambda: qtg.QDesktopServices.openUrl(qtc.QUrl("https://github.com/etnorth/BaSSET")))
        help_menu.addAction(github_button)

        self.about_window = None
        about_button = qtg.QAction(qtg.QIcon(f"{self.configpath}/assets/icon.png"), "About", self)
        about_button.triggered.connect(self.about_button_clicked)
        help_menu.addAction(about_button)

        if os.path.exists(self.configfile):
            self.read_config_file()
            self.update_config_file()

        self.display_algorithm_widgets()
        self.display_reconstruction_widgets()

    def close_event(self, event):
        qtw.QApplication.quit()

    def about_button_clicked(self):
        if self.about_window is None:
            self.about_window = AboutDialog(self)
            self.about_window.show()
        else:
            self.about_window.close()
            self.about_window = None

    def display_reconstruction_widgets(self):
        """Displays widgets for reconstructions equaling number of components, and hides the rest"""
        for widget in self.reconstruct_widgets[:self.numComponentsSlider.value()]:
            widget.show()
        for widget in self.reconstruct_widgets[self.numComponentsSlider.value():]:
            widget.hide()

    def set_indir(self):
        if self.indirLabel.text() == "Select the folder containing your data files":
            indir = str(qtw.QFileDialog.getExistingDirectory(self))
        else:
            indir = str(qtw.QFileDialog.getExistingDirectory(self, directory=os.path.dirname(self.indirLabel.text())))
        if indir == "": # If the operation was cancelled
            return None
        self.indirLabel.setText(indir)
        self.add_recentdir(indir)

    def add_recentdir(self, recentdir):
        recentdir_button = qtg.QAction(recentdir, self)
        recentdir_button.triggered.connect(lambda: self.indirLabel.setText(recentdir_button.text()))
        recentdir_button.triggered.connect(lambda: self.add_recentdir(recentdir))
        if recentdir in (action.text() for action in self.file_submenu_recent.actions()):
            self.file_submenu_recent.removeAction(next(action for action in self.file_submenu_recent.actions() if action.text()==recentdir))
        elif len(self.file_submenu_recent.actions()) >= 10+2:
            self.file_submenu_recent.removeAction(self.file_submenu_recent.actions()[-3]) # -1 is "clear", -2 is separator, so -3 should be oldest "recent"
        elif len(self.file_submenu_recent.actions()) == 2: # To ensure first "recent" is above the separator
            self.file_submenu_recent.insertAction(self.file_submenu_recent_separator, recentdir_button)
            return None
        self.file_submenu_recent.insertAction(self.file_submenu_recent.actions()[0], recentdir_button)

    def display_algorithm_widgets(self):
        for widget in (self.PCAalgorithmWidgets +
                       self.NMFalgorithmWidgets +
                       self.ICAalgorithmWidgets +
                       self.SNMFalgorithmWidgets):
                    widget.hide()
        match self.algorithmGroup.checkedButton().text():
            case "PCA":
                """for widget in self.PCAalgorithmWidgets:
                    widget.show()"""
                self.PCAwhitenCheck.show()
                self.PCAsolverDropdown.show()
                self.PCAtolSpinbox.show() if self.PCAsolverDropdown.currentText=='arpack' else self.PCAtolSpinbox.hide()
                self.PCAiterated_powerAutoCheckbox.show() if self.PCAsolverDropdown.currentText=='randomized' else self.PCAiterated_powerAutoCheckbox.hide()
                self.PCAiterated_powerSpinbox.show() if self.PCAsolverDropdown.currentText=='randomized' else self.PCAiterated_powerSpinbox.hide()
                self.PCAn_oversampledSpinbox.show() if self.PCAsolverDropdown.currentText=='randomized' else self.PCAn_oversampledSpinbox.hide()
                self.PCApower_iteration_normalizerDropdown.show() if self.PCAsolverDropdown.currentText=='randomized' else self.PCApower_iteration_normalizerDropdown.hide()
            case "NMF":
                for widget in self.NMFalgorithmWidgets:
                    widget.show()
                self.NMFbeta_lossDropdown.show() if self.NMFsolverDropdown.currentText=='mu' else self.NMFbeta_lossDropdown.hide()
                self.NMFl1_ratioDSpinbox.show() if self.NMFsolverDropdown.currentText=='cd' else self.NMFl1_ratioDSpinbox.show()
            case "ICA":
                for widget in self.ICAalgorithmWidgets:
                    widget.show()
            case "SNMF":
                for widget in self.SNMFalgorithmWidgets:
                    widget.show()

    def read_config_file(self):
        with open(self.configfile, encoding='utf-8') as infile:
            lines = infile.readlines()
            for i, line in enumerate(lines):
                line = line.replace('\n','')
                pairs = line.split(": ")
                name = pairs[0]
                value = pairs[1]
                if value != "":
                    match name:
                        case "Current file directory":
                            self.indirLabel.setText(value)
                        case "Recent directories":
                            for dir in reversed(value.split(", ")):
                                if dir != "":
                                    self.add_recentdir(dir)
                        case "Input format":
                            for button in self.inputformatGroup.buttons():
                                if value == button.text():
                                    button.setChecked(True)
                            if value not in [button.text() for button in self.inputformatGroup.buttons()]:
                                raise ValueError(f"Did not recognize \"{value}\" as an input format")
                        case "Wavelength":
                            self.wavelengthWidget.blockSignals(True) # blockSignals ensures .setValue doesn't trigger valeuChanged and updates the config, overwriting the load file
                            self.wavelengthWidget.setValue(float(value))
                            self.wavelengthWidget.blockSignals(False)
                        case "Algorithm":
                            for button in self.algorithmGroup.buttons():
                                if value == button.text():
                                    button.setChecked(True)
                            if value not in [button.text() for button in self.algorithmGroup.buttons()]:
                                raise ValueError(f"Did not recognize \"{value}\" as an algorithm")
                        case "Number of components":
                            self.numComponentsSlider.blockSignals(True) # blockSignals ensures .setValue doesn't trigger valeuChanged and updates the config, overwriting the load file
                            self.numComponentsSlider.setValue(int(value))
                            self.numComponentsLabel.setText(value)
                            self.numComponentsSlider.blockSignals(False)
                        case _:
                            print(f"\"{line}\" could not be parsed")

    def update_config_file(self):
        with open(self.configfile, "w", encoding='utf-8') as outfile:
            outfile.write(f"Current file directory: {self.indirLabel.text() if self.indirLabel.text()!='Select the folder containing your data files' else ''}\n")
            outfile.write(f"Recent directories: {', '.join(action.text() for action in self.file_submenu_recent.actions()[:-2])}\n")
            outfile.write(f"Input format: {self.inputformatGroup.checkedButton().text()}\n")
            outfile.write(f"Wavelength: {self.wavelengthWidget.value():.6f}\n")
            outfile.write(f"Algorithm: {self.algorithmGroup.checkedButton().text()}\n")
            outfile.write(f"Number of components: {self.numComponentsSlider.value()}\n")

    def run_analysis(self):
        print("Beginning analysis...")
        angles, intensities = import_dataset(self.indirLabel.text() + os.path.sep)
        numComponents = self.numComponentsSlider.value()

        if self.convert2QCheckbox.isChecked():
            angles = theta_to_Q(angles, self.wavelengthWidget.value())

        match self.algorithmGroup.checkedButton().text():
            case "PCA":
                fitted, transformed, reconstructed = PCA_analysis(intensities, numComponents,
                                                                  whiten=self.PCAwhitenCheck.isChecked(),
                                                                  svd_solver=self.PCAsolverDropdown.currentText(),
                                                                  tol=10**(self.PCAtolSpinbox.value()),
                                                                  iterated_power='auto' if self.PCAiterated_powerAutoCheckbox.isChecked else self.PCAiterated_powerSpinbox.value(),
                                                                  n_oversamples=self.PCAn_oversampledSpinbox.value(),
                                                                  power_iteration_normalizer=self.PCApower_iteration_normalizerDropdown.currentText())
                errors = None
            case "NMF":
                fitted, transformed, reconstructed, errors = NMF_analysis(intensities, numComponents,
                                                                          init=self.NMFinitDropdown.currentText(),
                                                                          solver=self.NMFsolverDropdown.currentText(),
                                                                          beta_loss=self.NMFbeta_lossDropdown.currentText(),
                                                                          tol=10**(self.NMFtolSpinbox.value()),
                                                                          max_iter=self.NMFmax_iterSpinbox.value(),
                                                                          alpha_W=self.NMFalpha_WDSpinbox.value(),
                                                                          alpha_H='same' if self.NMFalpha_HsameCheckbox.isChecked() else self.NMFalpha_HDSpinbox.value(),
                                                                          l1_ratio=self.NMFl1_ratioDSpinbox.value())
            case "ICA":
                fitted, transformed, reconstructed = ICA_analysis(intensities, numComponents,
                                                                  algorithm=self.ICAalgorithmDropdown.currentText(),
                                                                  whiten=False if self.ICAwhitenDropdown.currentText()=='False' else self.ICAwhitenDropdown.currentText(),
                                                                  fun=self.ICAfunDropdown.currentText(),
                                                                  max_iter=self.ICAmax_iterSpinbox.value(),
                                                                  tol=10**(self.ICAtolSpinbox.value()),
                                                                  whiten_solver=self.ICAwhiten_solverDropdown.currentText())
                errors = None
            case "SNMF":
                fitted, transformed, scores, stretch, reconstructed = SNMF_analysis(intensities, numComponents,
                                                                               min_iter=self.SNMFmin_iterSpinbox.value(),
                                                                               max_iter=self.SNMFmax_iterSpinbox.value(),
                                                                               tol=self.SNMFtol_SciSpinbox.value(),
                                                                               rho=self.SNMFrho_SciSpinbox.value(),
                                                                               eta=self.SNMFeta_DSpinbox.value())

        print("Analysis completed")
        self.plot_analysis(angles, intensities, numComponents, fitted, transformed, reconstructed, errors)

    def plot_analysis(self, angles, intensities, numComponents, fitted, transformed, reconstructed, errors=None):
        if numComponents < 3:
            plotwidth = 3
        else:
            plotwidth = numComponents

        if self.convert2QCheckbox.isChecked():
            xlabel = "Q (Å⁻¹)"
        else:
            xlabel = self.inputformatGroup.checkedButton().text()

        reconNum = []
        for widget in self.reconstruct_widgets:
            if widget.isVisible():
                reconNum.append(widget.value()-1)
        if any(x>len(reconstructed) for x in reconNum):
            print("Reconstruction scan larger than numer of scan. Defaulting to uniform distribution")
            reconNum[0]=-1

        if any(x==-1 for x in reconNum): # If any reconstuction widgets are 0, meaning -1 due to the line above
            reconNum = list(np.linspace(0, len(reconstructed), numComponents-1, endpoint=False, dtype=int))
            reconNum.append(len(reconstructed)-1)  # By using numComponents-1 and appending the end-point we get first and last scan

        colors = uio_cmp(np.linspace(0, 1, numComponents))

        fig, axs = plt.subplots(3, plotwidth, layout="constrained", )#, gridspec_kw = {"hspace":0.1})
        for i in range(numComponents):
            axs[0][i].plot(angles[i], fitted.components_[i], "k")
            axs[0][i].set_title(f"Component {i+1}")
            axs[0][i].set_xlabel(xlabel)
            axs[0][i].set_xlim(self.plot_xmin_DSpinBox.value(), min((self.plot_xmax_DSpinBox.value(), max(angles[i]))))
            axs[0][i].set_ylim(min(fitted.components_[i])-max(fitted.components_[i])*0.025, max(fitted.components_[i])*1.025)

            axs[1][i].plot(angles[reconNum[i]], reconstructed[reconNum[i]], color="#DD0000", label="Reconstructed")
            axs[1][i].plot(angles[reconNum[i]], intensities[reconNum[i]], "k:", label="Original")
            axs[1][i].plot(angles[reconNum[i]], intensities[reconNum[i]]-reconstructed[reconNum[i]], color="#2EC483", label="Difference")
            axs[1][i].ticklabel_format(axis="y", style="sci", scilimits=[0,0])
            axs[1][i].set_title(f"Scan {reconNum[i]+1}") # Reconstruction num
            axs[1][i].set_xlabel(xlabel)
            axs[1][i].sharex(axs[0][i])
            #axs[1][i].set_xlim(self.plot_xmin_DSpinBox.value(), min((self.plot_xmax_DSpinBox.value(), max(angles[i]))))
            axs[1][i].set_ylim(min(intensities[reconNum[i]])-max(intensities[reconNum[i]])*0.025, max(intensities[reconNum[i]])*1.025)

            axs[2][0].plot(np.arange(1,len(angles)+1), transformed[:,i], color=colors[i], label=f"{i+1}")
        
        for i in range(numComponents-1):
            axs[0][i].sharex(axs[0][i+1])

        axs[1][numComponents-1].legend()

        for i in range(len(reconNum)):
            axs[2][0].axvline(reconNum[i]+1, color="k", linestyle=":", label="Scans" if i==0 else '')
        axs[2][0].ticklabel_format(axis="y", style="sci", scilimits=[0,0])
        axs[2][0].legend()
        axs[2][0].set_title("Scores")

        match self.algorithmGroup.checkedButton().text():
            case "PCA":
                axs[2][1].plot(np.arange(1, min(10,min(np.shape(intensities)))+1), fitted.explained_variance_ratio_*100, "ko-")
                axs[2][1].set_title("Explained variances")
            case "NMF":
                axs[2][1].plot(np.arange(1, min(10,min(np.shape(intensities)))+1), errors, "ko-")
                axs[2][1].set_title("Reconstruction error")
            case "SNMF":
                axs[2][1].plot(np.arange(1, min(10,min(np.shape(intensities)))+1), errors, "ko-")
                axs[2][1].set_title("Reconstruction error (Not Implemented)")
        axs[2][1].set_xlabel("# of Components")

        fig.canvas.manager.set_window_title(f"{numComponents} component {self.algorithmGroup.checkedButton().text()} on {self.indirLabel.text().split('/')[-1]}")

        # Maximize window
        manager = plt.get_current_fig_manager()
        try:
            manager.window.showMaximized() # For Qt
        except AttributeError:
            try:
                manager.frame.Maximize(True) # For Wx
            except AttributeError:
                manager.full_screen_toggle() # Fallback for some backends

        fig.show()

        if self.export_results_checkbox.isChecked():
            print("Exporting results")
            self.export_results(angles, intensities, fitted, transformed, reconstructed, fig, errors)

    def write_summary(self, results_path, errors=None):
        with open(f"{results_path}/summary.txt", "w") as outfile:
            outfile.write(f"Summary of {self.numComponentsSlider.value()} component {self.algorithmGroup.checkedButton().text()} analysis of data in \"...{self.indirLabel.text()[-51:]}\"\n\n")
            outfile.write("The following algorithm parameters where used:\n")
            match self.algorithmGroup.checkedButton().text():
                case "PCA":
                    outfile.write(f"Whiten: {self.PCAwhitenCheck.isChecked()}\n")
                    outfile.write(f"SVD solver: {self.PCAsolverDropdown.currentText()}\n")
                    if self.PCAsolverDropdown.currentText()=="arpack": outfile.write(f"Tolerance: {10**self.PCAtolSpinbox.value()}\n")
                    if self.PCAsolverDropdown.currentText()=="randomized":
                        outfile.write(f"# of iterations (Power method): {self.PCAiterated_powerSpinbox() if self.PCAiterated_powerAutoCheckbox.isChecked() else 'auto'}\n")
                        outfile.write(f"Additional vectors to sample data: {self.PCAn_oversampledSpinbox.value()}\n")
                        outfile.write(f"Power iteration normalizer: {self.PCApower_iteration_normalizerDropdown.currentText()}\n")
                        outfile.write(f"\nThe PCA explained variance for {self.numComponentsSlider.value()} components was: {errors[self.numComponentsSlider.value()+1]}\n")
                case "NMF":
                    outfile.write(f"Initialization method: {self.NMFinitDropdown.currentText()}\n")
                    outfile.write(f"Numerical solver: {self.NMFsolverDropdown.currentText()}\n")
                    if self.NMFsolverDropdown.currentText()=='mu': outfile.write(f"Beta divergence to minimize: {self.NMFbeta_lossDropdown.currentText()}\n")
                    outfile.write(f"Tolerance: {10**self.NMFtolSpinbox.value()}\n")
                    outfile.write(f"Maximum number of iterations: {self.NMFmax_iterSpinbox.value()}\n")
                    outfile.write(f"Regularization constant for features: {self.NMFalpha_WDSpinbox.value()}\n")
                    outfile.write(f"Regularization constant for samples: {self.NMFalpha_HDSpinbox.value()}\n") # Should work even if alpha_Hsame=True
                    outfile.write(f"Regularization mixing parameter (0=l2, 1=l1): {self.NMFl1_ratioDSpinbox.value()}\n")
                    outfile.write(f"\nThe NMF reconstruction error for {self.numComponentsSlider.value()} components was: {errors[self.numComponentsSlider.value()+1]:.2f}\n")
                case "ICA":
                    outfile.write(f"Algorithm: {self.ICAalgorithmDropdown.currentText()}\n")
                    outfile.write(f"Whitening strategy: {self.ICAwhitenDropdown.currentText()}\n")
                    outfile.write(f"Whitening solver: {self.ICAwhiten_solverDropdown.currentText()}\n")
                    outfile.write(f"Functional G form function: {self.ICAfunDropdown.currentText()}\n")
                    outfile.write(f"Maximum number of iterations: {self.ICAmax_iterSpinbox.value()}\n")
                    outfile.write(f"Tolerance: {10**self.ICAtolSpinbox.value()}\n")
        print("Summary written")

    def write_components(self, results_path, angles, fitted):
        os.mkdir(f"{results_path}/components")
        for i, component in enumerate(fitted.components_):
            with open(f"{results_path}/components/component_{i+1:0{2}}.xy", "w") as outfile:
                for j in range(len(angles[i])):
                    outfile.write(f"{angles[i][j]}\t{component[j]}\n")
        print("Components written")

    def write_scores(self, results_path, transformed):
        with open(f"{results_path}/scores.csv", "w") as outfile:
            for i in range(self.numComponentsSlider.value()-1):
                outfile.write(f"Component {i+1:0{2}},")
            outfile.write(f"Component {self.numComponentsSlider.value():0{2}}\n")

            for i in range(len(transformed)):
                for j in range(len(transformed[0])-1):
                    outfile.write(f"{transformed[i][j]},")
                outfile.write(f"{transformed[i][len(transformed[0])-1]}\n")
        print("Scores written")

    def write_reconstructions(self, results_path, angles, intensities, reconstructed):
        os.mkdir(f"{results_path}/reconstructions")
        for i in range(len(reconstructed)):
            with open(f"{results_path}/reconstructions/recon_scan_{i+1:0{4}}.xy", "w") as outfile:
                for j in range(len(angles[i])):
                    outfile.write(f"{angles[i][j]}\t{reconstructed[i][j]}\n")
        print("Reconstructions written")
        for i in range(len(reconstructed)):
            with open(f"{results_path}/reconstructions/diff_scan_{i+1:0{4}}.xy", "w") as outfile:
                for j in range(len(angles[i])):
                    outfile.write(f"{angles[i][j]}\t{intensities[i][j]-reconstructed[i][j]}\n")
        print("Differences written")

    def export_results(self, angles, intensities, fitted, transformed, reconstructed, fig=None, errors=None):
        if not os.path.exists(f"{self.indirLabel.text()}/BaSSET_results"):
            os.mkdir(f"{self.indirLabel.text()}/BaSSET_results")

        export_time = datetime.now().strftime("%y%m%d-%H%M%S")
        results_path = f"{self.indirLabel.text()}/BaSSET_results/{export_time}_{self.algorithmGroup.checkedButton().text()}_{self.numComponentsSlider.value()}"
        os.mkdir(results_path)

        if fig!=None:
            fig.savefig(f"{results_path}/overview.jpg")
        self.write_summary(results_path, errors)
        self.write_components(results_path, angles, fitted)
        self.write_scores(results_path, transformed)
        self.write_reconstructions(results_path, angles, intensities, reconstructed)

class AboutDialog(qtw.QDialog):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("About BaSSET")
        self.setWindowIcon(qtg.QIcon(f"{parent.configpath}/assets/icon.png"))
        self.setWindowFlags(self.windowFlags() & ~qtc.Qt.WindowType.WindowContextHelpButtonHint)

        layout = qtw.QVBoxLayout()

        self.program = qtw.QLabel("<h1>BaSSET</h1>")
        self.program.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.program)

        self.logo = qtw.QLabel()
        self.logo.setPixmap(qtg.QPixmap(f"{parent.configpath}/assets/icon.png"))
        self.logo.setAlignment(qtc.Qt.AlignmentFlag.AlignTop | qtc.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.logo)

        """
        self.version = qtw.QLabel("Version: ALPHA_DEV")
        self.version.setAlignment(qtc.Qt.AlignCenter)
        layout.addWidget(self.version)
        """

        self.company = qtw.QLabel("Developed at NAFUMA Battery - University of Oslo")
        self.company.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.company.resize(self.company.sizeHint())
        layout.addWidget(self.company)

        self.funding = qtw.QLabel("Funded by the Research Council of Norway<br>"
        "(<a href='https://prosjektbanken.forskningsradet.no/en/project/FORISS/325316'>BaSSET 325316</a>)")
        self.funding.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        #self.funding.setTextFormat(qtc.QtRichText)
        self.funding.setTextInteractionFlags(qtc.Qt.TextInteractionFlag.TextBrowserInteraction)
        self.funding.setOpenExternalLinks(True)
        layout.addWidget(self.funding)

        self.developer = qtw.QLabel("Developer: Eira T. North")
        self.developer.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.developer)

        self.setLayout(layout)

        self.setFixedSize(qtc.QSize(300, 400))

def main():
    print("Starting BaSSET...")
    app = qtw.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()

if __name__=="__main__":
    main()