import sys
import os
from glob import glob
from datetime import datetime

from natsort import natsorted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import PyQt5.QtCore as qtc
import PyQt5.QtWidgets as qtw
import PyQt5.QtGui as qtg

from sklearn.decomposition import PCA, NMF, FastICA
#from diffpy.stretched_nmf.snmf_class import SNMFOptimizer 

import platform
if platform.system() == "Windows":
    # Separates BaSSET from the "Pythonw.exe" ID so it can have its own tackbar icon
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u"UiO.BaSSET")

uio_palette = ['#86A4F7','#2EC483','#FB6666']
uio_cmp = LinearSegmentedColormap.from_list('UiO colourmap',uio_palette)

def import_dataset(dir, filetype, auto_filetype=False):
    """
    Imports files from chosen directory of chosen filetype
    """

    if auto_filetype: # Find most popular filetype in folder
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

    print(f"Looking for files in {dir} of type {filetype}")
    filenames = natsorted(glob(f"{dir}*{filetype}"))
    print(f"Found {len(filenames)} files of filetype {filetype}")
    dataset_dict = {} #np.empty(len(filenames), dtype=pd.DataFrame)

    
    if filetype=='.xy' or filetype=='.gr':
        names = ["angle", "intensity"]
    elif filetype=='.xye' or filetype=='.dat':
        names = ["angle", "intensity", "error"]
    else:
        print("The filetype is not supported! Yet..?")
        raise NotImplementedError

    for i, filename in enumerate(filenames):
            dataset_dict[filename] = pd.read_csv(filename, engine='python', sep=' +', names=names)

    print("Dataset inported")
    return dataset_dict

def theta_to_Q(angles, wavelength):
    return ((4*np.pi) / wavelength) * np.sin(np.deg2rad(angles)/2)

def PCA_analysis(angles, intensities, numComponents, whiten, svd_solver, tol, iterated_power, n_oversamples, power_iteration_normalizer):
    n_components = 10 # Setting this higher than the user's number ensures reporting of explained variances
    pca = PCA(n_components = n_components, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, n_oversamples=n_oversamples, power_iteration_normalizer=power_iteration_normalizer)
    X = pca.fit(intensities)
    transformed = pca.transform(intensities)
    reconstructed = pca.inverse_transform(transformed)

    return X, transformed, reconstructed

def NMF_analysis(angles, intensities, numComponents, init, solver, beta_loss, tol, max_iter, alpha_W, alpha_H, l1_ratio):
    n_components_list = np.arange(1, 10+1, dtype=int)
    errors = np.empty(len(n_components_list))
    X = None
    transformed = None
    reconstructed = None

    for i, n_components in enumerate(n_components_list):
        nmf = NMF(n_components=n_components, init=init, solver=solver, beta_loss=beta_loss, tol=tol, max_iter=max_iter, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio)
        X_temp = nmf.fit(intensities)
        errors[i] = X_temp.reconstruction_err_
        if numComponents == n_components:
            X = X_temp
            transformed = nmf.transform(intensities)
            reconstructed = nmf.inverse_transform(transformed)

    return X, transformed, reconstructed, errors

def ICA_analysis(angles, intensities, numComponents, algorithm, whiten, fun, max_iter, tol, whiten_solver):
    ica = FastICA(n_components = numComponents, algorithm=algorithm, whiten=whiten, fun=fun, max_iter=max_iter, tol=tol, whiten_solver=whiten_solver)
    X = ica.fit(intensities)
    transformed = ica.transform(intensities)
    reconstructed = ica.inverse_transform(transformed)

    return X, transformed, reconstructed

def SNMF_analysis(angles, intensities, numComponents, min_iter, max_iter, tol, rho, eta):
    NotImplemented
    """
    snmf = SNMFOptimizer(source_matrix = intensities, n_components = numComponents, min_iter=min_iter, max_iter=max_iter, tol=tol)
    X = snmf.fit(rho=rho, eta=eta)
    transformed = snmf.apply_transformation_matrix()
    reconstructed = snmf.reconstruct_matrix()

    return X, transformed, reconstructed
    """

class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        self.configpath = os.path.dirname(os.path.realpath(__file__))
        self.configfile = f"{self.configpath}/config.dat"

        self.about_window = None

        self.setWindowIcon(qtg.QIcon(f"{self.configpath}/assets/icon.ico"))
        self.setWindowTitle("Battery Signal Separation and Enhancement Toolbox")
        self.resize(qtc.QSize(460, 360)) # Increase to 640x480 in future?
        self.centralWidget = qtw.QWidget(self)
        self.setCentralWidget(self.centralWidget)

        ##############################
        ##### Input data widgets #####
        ##############################
        self.indirLabel = qtw.QLabel("Select the folder containing your data files", self.centralWidget)
        self.indirLabel.setAlignment(qtc.Qt.AlignRight)
        self.indirLabel.setGeometry(10, 10, 350, 20)
        self.indirLabel.setFrameShape(qtw.QFrame.Panel)
        self.indirLabel.setFrameShadow(qtw.QFrame.Sunken)

        self.indirButton = qtw.QPushButton("File directory", self.centralWidget)
        self.indirButton.setGeometry(370, 8, 80, 20)

        self.inputformatGroup = qtw.QButtonGroup(self.centralWidget)
        self.inputformatTitle = qtw.QLabel("Input format", self.centralWidget)
        self.inputformatTitle.setGeometry(35, 40, 70, 20)

        self.qButton = qtw.QRadioButton("Q (Å⁻¹)", self.centralWidget)
        self.qButton.setGeometry(10, 60, 55, 20)
        self.qButton.setChecked(True)

        self.rButton = qtw.QRadioButton("r (Å)", self.centralWidget)
        self.rButton.setGeometry(75, 60, 50, 20)
        self.rButton.setDisabled(True)

        self.thetaButton = qtw.QRadioButton("2θ (°)", self.centralWidget)
        self.thetaButton.setGeometry(10, 80, 55, 20)
        self.thetaButton.setToolTip("Uses supplied wavelength to display data in Q (Å⁻¹)")

        self.inputformatGroup.addButton(self.qButton)
        self.inputformatGroup.addButton(self.thetaButton)
        self.inputformatGroup.addButton(self.rButton)

        self.wavelengthWidget = qtw.QDoubleSpinBox(self.centralWidget, decimals=6)
        self.wavelengthWidget.setGeometry(60, 80, 80, 20)
        self.wavelengthWidget.setMinimum(0)
        self.wavelengthWidget.setSingleStep(0.000001)
        self.wavelengthWidget.setSuffix(' Å')

        self.filetypeGroup = qtw.QButtonGroup(self.centralWidget)
        self.filetypeTitle = qtw.QLabel("Filetype", self.centralWidget)
        self.filetypeTitle.setGeometry(40, 110, 50, 20)

        self.xyButton = qtw.QRadioButton(".xy", self.centralWidget)
        self.xyButton.setGeometry(10, 130, 40, 20)
        self.xyButton.setChecked(True)
        self.xyButton.setToolTip("XRD: Two-column plain text file")

        self.xyeButton = qtw.QRadioButton(".xye", self.centralWidget)
        self.xyeButton.setGeometry(60, 130, 40, 20)
        self.xyeButton.setToolTip("XRD: three-column plain text file")

        self.datButton = qtw.QRadioButton(".dat", self.centralWidget)
        self.datButton.setGeometry(10, 150, 40, 20)
        self.datButton.setToolTip("XRD: three-column plain text file")

        self.grButton = qtw.QRadioButton(".gr", self.centralWidget)
        self.grButton.setGeometry(60, 150, 40, 20)
        self.grButton.setToolTip("PDF: two-column plain text file")

        self.filetypeGroup.addButton(self.xyButton)
        self.filetypeGroup.addButton(self.xyButton)
        self.filetypeGroup.addButton(self.xyeButton)
        self.filetypeGroup.addButton(self.datButton)
        self.filetypeGroup.addButton(self.grButton)

        self.autofiletypeCheck = qtw.QCheckBox("auto", self.centralWidget)
        self.autofiletypeCheck.setGeometry(35, 170, 40, 20)
        self.autofiletypeCheck.setToolTip("Selects the most popular filetype in the chosen directory")

        #############################
        ##### Algorithm widgets #####
        #############################
        self.algorithmGroup = qtw.QButtonGroup(self.centralWidget)
        self.algorithmTitle = qtw.QLabel("Algorithm", self.centralWidget)
        self.algorithmTitle.setGeometry(220, 40, 50, 20)

        self.PCAButton = qtw.QRadioButton("PCA", self.centralWidget)
        self.PCAButton.setGeometry(170, 60, 40, 20)
        self.PCAButton.setChecked(True)
        self.PCAButton.setToolTip("Principal Component Analysis (scikit-learn)")

        self.NMFButton = qtw.QRadioButton("NMF", self.centralWidget)
        self.NMFButton.setGeometry(220, 60, 40, 20)
        self.NMFButton.setToolTip("Non-Negative Matrix Factorization (scikit-learn)")

        self.ICAButton = qtw.QRadioButton("ICA", self.centralWidget)
        self.ICAButton.setGeometry(270, 60, 40, 20)
        self.ICAButton.setToolTip("Independent Component Analysis (scikit-learn)")

        self.SNMFButton = qtw.QRadioButton("SNMF", self.centralWidget)
        self.SNMFButton.setGeometry(170, 80, 50, 20)
        self.SNMFButton.setDisabled(True)
        self.SNMFButton.setToolTip("Stretched Non-Negative Matrix Factorization (diffpy)")

        self.algorithmGroup.addButton(self.PCAButton)
        self.algorithmGroup.addButton(self.NMFButton)
        self.algorithmGroup.addButton(self.ICAButton)
        self.algorithmGroup.addButton(self.SNMFButton)


        self.numComponentsTitle = qtw.QLabel("Number of Components", self.centralWidget)
        self.numComponentsTitle.setGeometry(185, 110, 120, 20)

        self.numComponentsSlider = qtw.QSlider(qtc.Qt.Horizontal, self.centralWidget)
        self.numComponentsSlider.setTickPosition(qtw.QSlider.TicksAbove | qtw.QSlider.TicksBelow)
        self.numComponentsSlider.setTickInterval(1)
        self.numComponentsSlider.setGeometry(190, 130, 90, 20)
        self.numComponentsSlider.setMinimum(2)
        self.numComponentsSlider.setMaximum(10)
        self.numComponentsSlider.setValue(4)
        self.numComponentsSlider.setSingleStep(1)
        self.numComponentsSlider.setPageStep(1)

        self.numComponentsLabel = qtw.QLabel(str(self.numComponentsSlider.value()), self.centralWidget)
        self.numComponentsLabel.setAlignment(qtc.Qt.AlignRight)
        self.numComponentsLabel.setGeometry(280, 130, 12, 20)


        self.algorithmParametersTitle = qtw.QLabel("Algorithm Parameters", self.centralWidget)
        self.algorithmParametersTitle.setGeometry(190, 160, 110, 20)

        ##########################
        ##### PCA parameters #####
        ##########################
        self.PCAwhitenCheck = qtw.QCheckBox("Whiten", self.centralWidget)
        self.PCAwhitenCheck.setGeometry(130, 180, 55, 20)
        self.PCAwhitenCheck.setToolTip("Whitening will remove some information from the transformed signal\n" \
        "(the relative variance scales of the components) but can sometime\n" \
        "improve the predictive accuracy of the downstream estimators\n" \
        "by making their data respect some hard-wired assumptions")

        self.PCAsolverDropdown = qtw.QComboBox(self.centralWidget)
        self.PCAsolverDropdown.addItems(['auto', 'full', 'covariance_eigh', 'arpack', 'randomized'])
        self.PCAsolverDropdown.setGeometry(190, 180, 105, 20)
        self.PCAsolverDropdown.setToolTip("The type of Singular Value Decomposition solver to use:\n" \
        "auto (default): Chooses solver based on size of dataset and number of components\n" \
        "full: Runs exact full SVD\n" \
        "covariance_eigh: Precomputes covarience for eigenvalue decompositon.\n" \
        "    Efficient for many scans of few datapoints (rare for scattering)\n" \
        "arpack: Runs SVD truncated to number of components.\n" \
        "    Requires fewer components than number of scans\n" \
        "randomized: Runs randomized SVD")

        # Only for 'arpack' solver
        self.PCAtolSpinbox = qtw.QSpinBox(self.centralWidget)
        self.PCAtolSpinbox.setPrefix("1e")
        self.PCAtolSpinbox.setMinimum(-20)
        self.PCAtolSpinbox.setMaximum(0)
        self.PCAtolSpinbox.setValue(0)
        self.PCAtolSpinbox.setSingleStep(1)
        self.PCAtolSpinbox.setGeometry(130, 205, 50, 20)
        self.PCAtolSpinbox.setToolTip("Tolerance for singular values using 'arpack'")
        self.PCAsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.PCAtolSpinbox.show()
            if currentText=='arpack'
            else self.PCAtolSpinbox.hide()
        )

        # Only for 'randomized' solver
        self.PCAiterated_powerSpinbox = qtw.QSpinBox(self.centralWidget)
        self.PCAiterated_powerSpinbox.setMinimum(0)
        self.PCAiterated_powerSpinbox.setMaximum(999999)
        self.PCAiterated_powerSpinbox.setValue(0)
        self.PCAiterated_powerSpinbox.setSingleStep(100)
        self.PCAiterated_powerSpinbox.setGeometry(130, 205, 60, 20)
        self.PCAiterated_powerSpinbox.setToolTip("Number of iterations for the power method in 'randomized'")
        self.PCAsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.PCAiterated_powerSpinbox.show()
            if currentText=='randomized'
            else self.PCAiterated_powerSpinbox.hide()
        )

        # Only for 'randomized' solver
        self.PCAiterated_powerAutoCheckbox = qtw.QCheckBox("auto", self.centralWidget)
        self.PCAiterated_powerAutoCheckbox.setGeometry(130, 230, 40, 20)
        self.PCAiterated_powerAutoCheckbox.setToolTip("Automatically choose number of iterations")
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
        self.PCAn_oversampledSpinbox = qtw.QSpinBox(self.centralWidget)
        self.PCAn_oversampledSpinbox.setMinimum(0)
        self.PCAn_oversampledSpinbox.setMaximum(50)
        self.PCAn_oversampledSpinbox.setValue(10)
        self.PCAn_oversampledSpinbox.setSingleStep(1)
        self.PCAn_oversampledSpinbox.setGeometry(195, 205, 40, 20)
        self.PCAn_oversampledSpinbox.setToolTip("Additional number of random vectors to sample using 'randomized'")
        self.PCAsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.PCAn_oversampledSpinbox.show()
            if currentText=='randomized'
            else self.PCAn_oversampledSpinbox.hide()
        )

        # Only for 'randomized' solver // Not for 'arpack' solver
        self.PCApower_iteration_normalizerDropdown = qtw.QComboBox(self.centralWidget)
        self.PCApower_iteration_normalizerDropdown.addItems(['auto', 'QR', 'LU', 'none'])
        self.PCApower_iteration_normalizerDropdown.setGeometry(240, 205, 50, 20)
        self.PCApower_iteration_normalizerDropdown.setToolTip("Power iteration normalizer using 'randomized'")
        self.PCAsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.PCApower_iteration_normalizerDropdown.show()
            if currentText=='randomized'
            else self.PCApower_iteration_normalizerDropdown.hide()
        )

        # Ignored PCA parameters:
        # random_state

        ##########################
        ##### NMF parameters #####
        ##########################
        self.NMFinitDropdown = qtw.QComboBox(self.centralWidget)
        self.NMFinitDropdown.addItems(["nndsvda", "random", "nndsvd", "nnsvdar"])
        self.NMFinitDropdown.setGeometry(130, 180, 65, 20)
        self.NMFinitDropdown.setToolTip("Method used to initialize the procedure:\n" \
        "('nndsvda' is reccommended for XRD and 'nndsvd' is recommended for PDF)\n" \
        "nndsvda (default): Better when sparsity is not desired\n" \
        "random: Random non-negative matrices\n" \
        "nndsvd: Non-negative Double Singular Value Decomposition (better for sparseness)\n" \
        "nnsvdar: Faster, less accurate alternative to NNDSVDa when sparsity is not desired")

        self.NMFsolverDropdown = qtw.QComboBox(self.centralWidget)
        self.NMFsolverDropdown.addItems(["cd", "mu"])
        self.NMFsolverDropdown.setGeometry(200, 180, 40, 20)
        self.NMFsolverDropdown.setToolTip("Numerical solver to use:\n" \
        "cd (default): Coordinate Descent\n" \
        "mu: Multiplicative Update")

        # Only for 'mu' solver 
        self.NMFbeta_lossDropdown = qtw.QComboBox(self.centralWidget)
        self.NMFbeta_lossDropdown.addItems(["frobenius", "kullback-leibler"]) # "itakura-saito" not included as it cannot have zeros in input data, which XRD/PDF often can have
        self.NMFbeta_lossDropdown.setGeometry(245, 180, 95, 20)
        self.NMFbeta_lossDropdown.setToolTip("Beta divergence to be minimized," \
        "measuring the distance between X and the dot product WH using 'mu':\n" \
        "frobenius is default")
        self.NMFsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.NMFbeta_lossDropdown.show()
            if currentText=='mu'
            else self.NMFbeta_lossDropdown.hide()
        )

        self.NMFtolSpinbox = qtw.QSpinBox(self.centralWidget)
        self.NMFtolSpinbox.setPrefix("1e")
        self.NMFtolSpinbox.setMinimum(-20)
        self.NMFtolSpinbox.setMaximum(0)
        self.NMFtolSpinbox.setValue(-4)
        self.NMFtolSpinbox.setSingleStep(1)
        self.NMFtolSpinbox.setGeometry(130, 205, 55, 20)
        self.NMFtolSpinbox.setToolTip("Tolerance of the stopping condition")

        self.NMFmax_iterSpinbox = qtw.QSpinBox(self.centralWidget)
        self.NMFmax_iterSpinbox.setMinimum(0)
        self.NMFmax_iterSpinbox.setMaximum(5000)
        self.NMFmax_iterSpinbox.setValue(2500)
        self.NMFmax_iterSpinbox.setGeometry(190, 205, 50, 20)
        self.NMFmax_iterSpinbox.setToolTip("Maximum number of iterations before timing out")

        self.NMFalpha_WDSpinbox = qtw.QDoubleSpinBox(self.centralWidget)
        self.NMFalpha_WDSpinbox.setMinimum(0)
        self.NMFalpha_WDSpinbox.setMaximum(10)
        self.NMFalpha_WDSpinbox.setValue(0)
        self.NMFalpha_WDSpinbox.setDecimals(5)
        self.NMFalpha_WDSpinbox.setSingleStep(0.00005)
        self.NMFalpha_WDSpinbox.setGeometry(130, 230, 65, 20)
        self.NMFalpha_WDSpinbox.setToolTip("Constant that multiplies the regularization terms of the features:\n" \
        "(Regularization a penalty term to constrain parameter complexity and reduce overfitting)\n" \
        "0 (default) means no regularization")

        self.NMFalpha_HDSpinbox = qtw.QDoubleSpinBox(self.centralWidget)
        self.NMFalpha_HDSpinbox.setMinimum(0)
        self.NMFalpha_HDSpinbox.setMaximum(10)
        self.NMFalpha_HDSpinbox.setValue(0)
        self.NMFalpha_HDSpinbox.setDecimals(5)
        self.NMFalpha_HDSpinbox.setSingleStep(0.00005)
        self.NMFalpha_HDSpinbox.setGeometry(200, 230, 65, 20)
        self.NMFalpha_HDSpinbox.setToolTip("Constant that multiplies the regularization terms of the mixing of features:\n" \
        "0 means no regularization")

        self.NMFalpha_HsameCheckbox = qtw.QCheckBox("same", self.centralWidget)
        self.NMFalpha_HsameCheckbox.setGeometry(270, 230, 45, 20)
        self.NMFalpha_HsameCheckbox.setToolTip("Use same regularization constant for features and mixing")
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

        # Used in 'cd' solver
        self.NMFl1_ratioDSpinbox = qtw.QDoubleSpinBox(self.centralWidget)
        self.NMFl1_ratioDSpinbox.setMinimum(0)
        self.NMFl1_ratioDSpinbox.setMaximum(1)
        self.NMFl1_ratioDSpinbox.setValue(0)
        self.NMFl1_ratioDSpinbox.setSingleStep(0.05)
        self.NMFl1_ratioDSpinbox.setGeometry(245, 180, 45, 20)
        self.NMFl1_ratioDSpinbox.setToolTip("Regularization mixing parameter:\n" \
        "(0, default): elementwise l2 penalty aka. Frobenius Norm\n"
        "(1): elementwwise l1 penalty (1)\n")
        self.NMFsolverDropdown.currentTextChanged.connect(
            lambda currentText: self.NMFl1_ratioDSpinbox.show()
            if currentText=='cd'
            else self.NMFl1_ratioDSpinbox.hide()
        )

        #Ignored NMF parmeters:
        # random_state
        # verbose
        # shuffle

        ##########################
        ##### ICA parameters #####
        ##########################
        self.ICAalgorithmDropdown = qtw.QComboBox(self.centralWidget)
        self.ICAalgorithmDropdown.addItems(['parallel', 'deflation'])
        self.ICAalgorithmDropdown.setGeometry(130, 180, 65, 20)
        self.ICAalgorithmDropdown.setToolTip("Specify which algorithm to use:\n" \
        "parallel is default")

        self.ICAwhitenDropdown = qtw.QComboBox(self.centralWidget)
        self.ICAwhitenDropdown.addItems(['unit-variance', 'arbitrary-variance', 'False'])
        self.ICAwhitenDropdown.setGeometry(200, 180, 110, 20)
        self.ICAwhitenDropdown.setToolTip("Whitening strategy to use. False means no whitening:\n" \
        "unit-variance (default): the whitening matrix is rescaled to ensure each recovered source has unit variance\n" \
        "arbitrary-variance: a whitening with variance arbitrary is used\n" \
        "False: Data is considered whitened and no whitening is performed")

        self.ICAfunDropdown = qtw.QComboBox(self.centralWidget)
        self.ICAfunDropdown.addItems(['logcosh', 'exp', 'cube'])
        self.ICAfunDropdown.setGeometry(245, 205, 60, 20)
        self.ICAfunDropdown.setToolTip("The functional form of the G function used in the approximation to neg-entropy:\n" \
        "logcosh is default")

        self.ICAmax_iterSpinbox = qtw.QSpinBox(self.centralWidget)
        self.ICAmax_iterSpinbox.setMinimum(0)
        self.ICAmax_iterSpinbox.setMaximum(5000)
        self.ICAmax_iterSpinbox.setValue(2500)
        self.ICAmax_iterSpinbox.setGeometry(130, 205, 50, 20)
        self.ICAmax_iterSpinbox.setToolTip("Maximum number of iterations during fit")

        self.ICAtolSpinbox = qtw.QSpinBox(self.centralWidget)
        self.ICAtolSpinbox.setPrefix("1e")
        self.ICAtolSpinbox.setMinimum(-20)
        self.ICAtolSpinbox.setMaximum(0)
        self.ICAtolSpinbox.setValue(-4)
        self.ICAtolSpinbox.setSingleStep(1)
        self.ICAtolSpinbox.setGeometry(185, 205, 55, 20)
        self.ICAtolSpinbox.setToolTip("A positive scalar giving the tolerance at which the un-mixing matrix is considered to have converged")

        self.ICAwhiten_solverDropdown = qtw.QComboBox(self.centralWidget)
        self.ICAwhiten_solverDropdown.addItems(['svd', 'eigh'])
        self.ICAwhiten_solverDropdown.setGeometry(130, 230, 45, 20)
        self.ICAwhiten_solverDropdown.setToolTip("The solver to use for whitening:\n" \
        "svd (default): More numerically stable if the problem is degenerate, and often faster for scattering\n" \
        "eigh: More memory efficient when there are more scans than datapoints (rare in scattering)")

        # Ignored ICA parameters:
        # fun_args
        # w_init
        # random_state

        ###########################
        ##### SNMF parameters #####
        ###########################
        """To be added"""
        
        #####################################
        ##### Analysis and plot widgets #####
        #####################################
        self.runAnalysisButton = qtw.QPushButton("Run analysis", self.centralWidget)
        self.runAnalysisButton.setGeometry(330, 40, 120, 40)

        self.plot_xmin_DSpinBox = qtw.QDoubleSpinBox(self.centralWidget)
        self.plot_xmin_DSpinBox.setMinimum(0)
        self.plot_xmin_DSpinBox.setMaximum(999)
        self.plot_xmin_DSpinBox.setValue(0)
        self.plot_xmin_DSpinBox.setDecimals(2)
        self.plot_xmin_DSpinBox.setSingleStep(0.01)
        self.plot_xmin_DSpinBox.setGeometry(330, 85, 60, 20)
        self.plot_xmin_DSpinBox.setToolTip("Minimum x-value in plots")

        self.plot_xmin_label = qtw.QLabel("X-axis min", self.centralWidget)
        self.plot_xmin_label.setGeometry(395, 85, 55, 20)

        self.plot_xmax_DSpinBox = qtw.QDoubleSpinBox(self.centralWidget)
        self.plot_xmax_DSpinBox.setMinimum(0)
        self.plot_xmax_DSpinBox.setMaximum(999)
        self.plot_xmax_DSpinBox.setValue(20)
        self.plot_xmax_DSpinBox.setDecimals(2)
        self.plot_xmax_DSpinBox.setSingleStep(0.01)
        self.plot_xmax_DSpinBox.setGeometry(330, 110, 60, 20)
        self.plot_xmax_DSpinBox.setToolTip("Maximum x-value in plots")
        
        self.plot_xmin_label = qtw.QLabel("X-axis max", self.centralWidget)
        self.plot_xmin_label.setGeometry(395, 110, 55, 20)

        self.export_results_checkbox = qtw.QCheckBox("Export results", self.centralWidget)
        self.export_results_checkbox.setGeometry(330, 135, 100, 20)

        ################################
        ##### Function connections #####
        ################################
        self.indirButton.clicked.connect(self.set_indir)
        self.indirButton.clicked.connect(self.update_config_file)
        self.inputformatGroup.buttonClicked.connect(self.update_config_file)
        self.wavelengthWidget.valueChanged.connect(self.update_config_file)
        self.filetypeGroup.buttonClicked.connect(self.update_config_file)
        self.autofiletypeCheck.clicked.connect(self.update_config_file)
        self.algorithmGroup.buttonClicked.connect(self.algorithm_widgets)
        self.algorithmGroup.buttonClicked.connect(self.update_config_file)
        self.numComponentsSlider.valueChanged.connect(lambda: self.numComponentsLabel.setText(str(self.numComponentsSlider.value())))
        self.numComponentsSlider.valueChanged.connect(self.update_config_file)
        self.runAnalysisButton.clicked.connect(self.run_analysis)

        ########################
        ##### Top bar menu #####
        ########################
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        new_indir_button = qtw.QAction("Set file directory...", self)
        new_indir_button.triggered.connect(self.set_indir)
        new_indir_button.triggered.connect(self.update_config_file)
        file_menu.addAction(new_indir_button)
        file_menu.addAction(new_indir_button)
        open_indir_button = qtw.QAction("Open file directory", self)
        open_indir_button.triggered.connect(lambda: qtg.QDesktopServices.openUrl(qtc.QUrl.fromLocalFile(self.indirLabel.text())))
        file_menu.addAction(open_indir_button)

        config_button = qtw.QAction("Open config file", self)
        config_button.triggered.connect(lambda: qtg.QDesktopServices.openUrl(qtc.QUrl.fromLocalFile(f"{self.configpath}/config.dat")))
        file_menu.addAction(config_button)
        
        file_menu.addSeparator()
        
        self.file_submenu_recent = file_menu.addMenu("Choose recent")
        
        self.file_submenu_recent_separator = self.file_submenu_recent.addSeparator()

        self.clear_recent_button = qtw.QAction("Clear recently opened", self)
        #self.clear_recent_button.triggered.connect(self.clear_recent_dirs)
        self.clear_recent_button.triggered.connect(lambda: [self.file_submenu_recent.removeAction(action) for action in self.file_submenu_recent.actions()[:-2]])
        self.clear_recent_button.triggered.connect(self.update_config_file)
        self.file_submenu_recent.addAction(self.clear_recent_button)

        file_menu.addSeparator()

        exit_button = qtw.QAction("Exit", self)
        exit_button.triggered.connect(self.close_event)
        file_menu.addAction(exit_button)
        

        help_menu = menu.addMenu("Help")
        github_button = qtw.QAction("Source code", self)
        github_button.triggered.connect(lambda: qtg.QDesktopServices.openUrl(qtc.QUrl("https://github.com/etnorth/BaSSET")))
        help_menu.addAction(github_button)

        about_button = qtw.QAction(qtg.QIcon(f"{self.configpath}/assets/icon.png"), "About", self)
        about_button.triggered.connect(self.about_button_clicked)
        help_menu.addAction(about_button)

        if os.path.exists(self.configfile):
            self.read_config_file()
            self.update_config_file()

        self.algorithm_widgets()

    def close_event(self, event):
        qtw.QApplication.quit()

    def about_button_clicked(self):
        if self.about_window is None:
            self.about_window = AboutDialog(self)
            self.about_window.show()
        else:
            self.about_window.close()
            self.about_window = None

    def set_indir(self):
        if self.indirLabel.text() == "Select the folder containing your data files":
            indir = str(qtw.QFileDialog.getExistingDirectory(self))
        else:
            indir = str(qtw.QFileDialog.getExistingDirectory(self, directory=os.path.dirname(self.indirLabel.text())))
        if indir == "": # If the operation was cancelled
            return None
        self.indirLabel.setText(indir)
        self.add_recentdir(indir)
        #import_data(self.indirLabel.text()+"\\", self.filetypeGroup.checkedButton().text())

    def add_recentdir(self, recentdir):
        recentdir_button = qtw.QAction(recentdir, self)
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

    def algorithm_widgets(self):

        match self.algorithmGroup.checkedButton().text():
            case "PCA":
                self.PCAwhitenCheck.show()
                self.PCAsolverDropdown.show()
                self.PCAtolSpinbox.show() if self.PCAsolverDropdown.currentText=='arpack' else self.PCAtolSpinbox.hide()
                self.PCAiterated_powerAutoCheckbox.show() if self.PCAsolverDropdown.currentText=='randomized' else self.PCAiterated_powerAutoCheckbox.hide()
                self.PCAiterated_powerSpinbox.show() if self.PCAsolverDropdown.currentText=='randomized' else self.PCAiterated_powerSpinbox.hide()
                self.PCAn_oversampledSpinbox.show() if self.PCAsolverDropdown.currentText=='randomized' else self.PCAn_oversampledSpinbox.hide()
                self.PCApower_iteration_normalizerDropdown.show() if self.PCAsolverDropdown.currentText=='randomized' else self.PCApower_iteration_normalizerDropdown.hide()

                self.NMFinitDropdown.hide()
                self.NMFsolverDropdown.hide()
                self.NMFbeta_lossDropdown.hide()
                self.NMFtolSpinbox.hide()
                self.NMFmax_iterSpinbox.hide()
                self.NMFalpha_WDSpinbox.hide()
                self.NMFalpha_HDSpinbox.hide()
                self.NMFalpha_HsameCheckbox.hide()
                self.NMFl1_ratioDSpinbox.hide()

                self.ICAalgorithmDropdown.hide()
                self.ICAwhitenDropdown.hide()
                self.ICAwhiten_solverDropdown.hide()
                self.ICAfunDropdown.hide()
                self.ICAmax_iterSpinbox.hide()
                self.ICAtolSpinbox.hide()
            case "NMF":
                self.PCAwhitenCheck.hide()
                self.PCAsolverDropdown.hide()
                self.PCAtolSpinbox.hide()
                self.PCAiterated_powerSpinbox.hide()
                self.PCAiterated_powerAutoCheckbox.hide()
                self.PCAn_oversampledSpinbox.hide()
                self.PCApower_iteration_normalizerDropdown.hide()

                self.NMFinitDropdown.show()
                self.NMFsolverDropdown.show()
                self.NMFbeta_lossDropdown.show() if self.NMFsolverDropdown.currentText=='mu' else self.NMFbeta_lossDropdown.hide()
                self.NMFtolSpinbox.show()
                self.NMFmax_iterSpinbox.show()
                self.NMFalpha_WDSpinbox.show()
                self.NMFalpha_HDSpinbox.show()
                self.NMFalpha_HsameCheckbox.show()
                self.NMFl1_ratioDSpinbox.show() if self.NMFsolverDropdown.currentText=='cd' else self.NMFl1_ratioDSpinbox.show()

                self.ICAalgorithmDropdown.hide()
                self.ICAwhitenDropdown.hide()
                self.ICAwhiten_solverDropdown.hide()
                self.ICAfunDropdown.hide()
                self.ICAmax_iterSpinbox.hide()
                self.ICAtolSpinbox.hide()
            case "ICA":
                self.PCAwhitenCheck.hide()
                self.PCAsolverDropdown.hide()
                self.PCAtolSpinbox.hide()
                self.PCAiterated_powerSpinbox.hide()
                self.PCAiterated_powerAutoCheckbox.hide()
                self.PCAn_oversampledSpinbox.hide()
                self.PCApower_iteration_normalizerDropdown.hide()

                self.NMFinitDropdown.hide()
                self.NMFsolverDropdown.hide()
                self.NMFbeta_lossDropdown.hide()
                self.NMFtolSpinbox.hide()
                self.NMFmax_iterSpinbox.hide()
                self.NMFalpha_WDSpinbox.hide()
                self.NMFalpha_HDSpinbox.hide()
                self.NMFalpha_HsameCheckbox.hide()
                self.NMFl1_ratioDSpinbox.hide()

                self.ICAalgorithmDropdown.show()
                self.ICAwhitenDropdown.show()
                self.ICAwhiten_solverDropdown.show()
                self.ICAfunDropdown.show()
                self.ICAmax_iterSpinbox.show()
                self.ICAtolSpinbox.show()

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
                        case "Data filetype":
                            for button in self.filetypeGroup.buttons():
                                if value == button.text():
                                    button.setChecked(True)
                            if value not in [button.text() for button in self.filetypeGroup.buttons()]:
                                raise ValueError(f"Did not recognize \"{value}\" as a data filetype")
                        case "Auto filetype":
                            if value=="True":
                                self.autofiletypeCheck.setChecked(True)
                            elif value=="False":
                                self.autofiletypeCheck.setChecked(False)
                            else:
                                raise ValueError(f"Did not recognize \"{value}\" as True or False")
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
                            print("\"{line}\" could not be parsed")

    def update_config_file(self):
        if self.autofiletypeCheck.isChecked():
            for widget in self.filetypeGroup.buttons():
                widget.setDisabled(True)
            self.qButton.setDisabled(False)
            self.thetaButton.setDisabled(False)
            self.rButton.setDisabled(False)
        else:
            for widget in self.filetypeGroup.buttons():
                widget.setDisabled(False)

        if self.filetypeGroup.checkedButton().text() == '.gr':
            self.qButton.setDisabled(True)
            self.thetaButton.setDisabled(True)
            self.rButton.setDisabled(False)
            self.rButton.setChecked(True)
        else:
            self.qButton.setDisabled(False)
            self.thetaButton.setDisabled(False)
            self.rButton.setDisabled(True)

        with open(self.configfile, "w", encoding='utf-8') as outfile:
            outfile.write(f"Current file directory: {self.indirLabel.text() if self.indirLabel.text()!='Select the folder containing your data files' else ''}\n")
            outfile.write(f"Recent directories: {', '.join(action.text() for action in self.file_submenu_recent.actions()[:-2])}\n")
            outfile.write(f"Input format: {self.inputformatGroup.checkedButton().text()}\n")
            outfile.write(f"Wavelength: {self.wavelengthWidget.value():.6f}\n")
            outfile.write(f"Data filetype: {self.filetypeGroup.checkedButton().text()}\n")
            outfile.write(f"Auto filetype: {self.autofiletypeCheck.isChecked()}\n")
            outfile.write(f"Algorithm: {self.algorithmGroup.checkedButton().text()}\n")
            outfile.write(f"Number of components: {self.numComponentsSlider.value()}\n")

    def run_analysis(self):
        if self.filetypeGroup.checkedButton().text() != '.gr' and self.inputformatGroup.checkedButton().text() == "r (Å)":
            print(f"The selected filetype ({self.filetypeGroup.checkedButton().text()}) is not compatible with the input format {self.inputformatGroup.checkedButton().text()}")
            return
        print("Beginning analysis...")
        dataset = import_dataset(self.indirLabel.text()+"\\", self.filetypeGroup.checkedButton().text(), self.autofiletypeCheck.isChecked())
        numComponents = self.numComponentsSlider.value()
        angles = np.array([values["angle"].to_numpy() for values in dataset.values()])
        intensities = np.array([values["intensity"].to_numpy() for values in dataset.values()])

        if self.inputformatGroup.checkedButton().text() == "2θ (°)":
            angles = theta_to_Q(angles, self.wavelengthWidget.value())

        match self.algorithmGroup.checkedButton().text():
            case "PCA":
                fitted, transformed, reconstructed = PCA_analysis(angles, intensities, numComponents,
                                                                  whiten=self.PCAwhitenCheck.isChecked(),
                                                                  svd_solver=self.PCAsolverDropdown.currentText(),
                                                                  tol=10**(self.PCAtolSpinbox.value()),
                                                                  iterated_power='auto' if self.PCAiterated_powerAutoCheckbox.isChecked else self.PCAiterated_powerSpinbox.value(),
                                                                  n_oversamples=self.PCAn_oversampledSpinbox.value(),
                                                                  power_iteration_normalizer=self.PCApower_iteration_normalizerDropdown.currentText())
                errors = None
            case "NMF":
                fitted, transformed, reconstructed, errors = NMF_analysis(angles, intensities, numComponents,
                                                                          init=self.NMFinitDropdown.currentText(),
                                                                          solver=self.NMFsolverDropdown.currentText(),
                                                                          beta_loss=self.NMFbeta_lossDropdown.currentText(),
                                                                          tol=10**(self.NMFtolSpinbox.value()),
                                                                          max_iter=self.NMFmax_iterSpinbox.value(),
                                                                          alpha_W=self.NMFalpha_WDSpinbox.value(),
                                                                          alpha_H='same' if self.NMFalpha_HsameCheckbox.isChecked() else self.NMFalpha_HDSpinbox.value(),
                                                                          l1_ratio=self.NMFl1_ratioDSpinbox.value())
            case "ICA":
                fitted, transformed, reconstructed = ICA_analysis(angles, intensities, numComponents,
                                                                  algorithm=self.ICAalgorithmDropdown.currentText(),
                                                                  whiten= False if self.ICAwhitenDropdown.currentText()=='False' else self.ICAwhitenDropdown.currentText(),
                                                                  fun=self.ICAfunDropdown.currentText(),
                                                                  max_iter=self.ICAmax_iterSpinbox.value(),
                                                                  tol=10**(self.ICAtolSpinbox.value()),
                                                                  whiten_solver=self.ICAwhiten_solverDropdown.currentText())
                errors = None
            case "SNMF":
                NotImplemented

        print("Analysis completed")
        self.plot_analysis(angles, intensities, numComponents, fitted, transformed, reconstructed, errors)

    def plot_analysis(self, angles, intensities, numComponents, fitted, transformed, reconstructed, errors=None):
        if numComponents < 3:
            plotwidth = 3
        else:
            plotwidth = numComponents

        if self.inputformatGroup.checkedButton().text() == "2θ (°)":
            xlabel = "Q (Å⁻¹)"
        else:
            xlabel = self.inputformatGroup.checkedButton().text()

        
        reconNum = list(np.linspace(0, len(reconstructed), numComponents-1, endpoint=False, dtype=int))
        reconNum.append(len(reconstructed)-1)  # By using numComponents-1 and appending the end-point we get first and last scan
        colors = uio_cmp(np.linspace(0, 1, numComponents))

        fig, axs = plt.subplots(3, plotwidth, layout="constrained")#, gridspec_kw = {"hspace":0.1})
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

        axs[2][0].ticklabel_format(axis="y", style="sci", scilimits=[0,0])
        axs[2][0].legend()
        axs[2][0].set_title("Scores")

        match self.algorithmGroup.checkedButton().text():
            case "PCA":
                axs[2][1].plot(np.arange(1, 10+1), fitted.explained_variance_ratio_*100, "ko-")
                axs[2][1].set_title("Explained variances")
            case "NMF":
                axs[2][1].plot(np.arange(1, 10+1), errors, "ko-")
                axs[2][1].set_title("Reconstruction error")
        axs[2][1].set_xlabel("# of Components")

        fig.canvas.manager.set_window_title(f"{numComponents} component {self.algorithmGroup.checkedButton().text()} on {self.indirLabel.text().split('/')[-1]}")

        fig.show()

        if self.export_results_checkbox.isChecked():
            print("Exporting results")
            self.export_results(angles, fitted, transformed, reconstructed, fig, errors)

    def write_summary(self, results_path, errors=None):
        with open(f"{results_path}/summary.txt", "w") as outfile:
            outfile.write(f"Summary of {self.numComponentsSlider.value()} component {self.algorithmGroup.checkedButton().text()} analysis of {self.filetypeGroup.checkedButton().text()[1:]} in ...{self.indirLabel.text()[-51:]}\n\n")
            outfile.write("The following algorithm parameters where used:\n")
            match self.algorithmGroup.checkedButton().text():
                case "PCA":
                    outfile.write(f"Whiten: {self.PCAwhitenCheck.isChecked()}\n")
                    outfile.write(f"SVD solver: {self.PCAsolverDropdown.currentText()}\n")
                    if self.PCAsolverDropdown.currentText()=="arpack": outfile.write(f"Tolerance: {self.PCAtolSpinbox.value()}\n")
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
                    outfile.write(f"\nThe NMF reconstruction error for {self.numComponentsSlider.value()} components was: {errors[self.numComponentsSlider.value()+1]}\n")
                case "ICA":
                    outfile.write(f"Algorithm: {self.ICAalgorithmDropdown.currentText()}\n")
                    outfile.write(f"Whitening strategy: {self.ICAwhitenDropdown.currentText()}\n")
                    outfile.write(f"Whitening solver: {self.ICAwhiten_solverDropdown.currentText()}\n")
                    outfile.write(f"Functional G form function: {self.ICAfunDropdown.currentText()}\n")
                    outfile.write(f"Maximum number of iterations: {self.ICAmax_iterSpinbox.value()}\n")
                    outfile.write(f"Tolerance: {10**self.ICAtolSpinbox.value()}\n")

    def write_components(self, results_path, angles, fitted):
        os.mkdir(f"{results_path}/components")
        match self.filetypeGroup.checkedButton().text():
            case ".xy" | ".gr":
                for i, component in enumerate(fitted.components_):
                    with open(f"{results_path}/components/component_{i+1}{self.filetypeGroup.checkedButton().text()}", "w") as outfile:
                        for j in range(len(angles[i])):
                            outfile.write(f"{angles[i][j]}\t{component[j]}\n")
            case ".xye" | ".dat":
                for i, component in enumerate(fitted.components_):
                        with open(f"{results_path}/components/component_{i+1}{self.filetypeGroup.checkedButton().text()}", "w") as outfile:
                            for j in range(len(angles[i])):
                                outfile.write(f"{angles[i][j]}\t{component[j]}\t0\n")

    def write_scores(self, results_path, transformed):
        with open(f"{results_path}/scores.csv", "w") as outfile:
            for i in range(self.numComponentsSlider.value()-1):
                outfile.write(f"Component {i+1},")
            outfile.write(f"Component {self.numComponentsSlider.value()}\n")

            for i in range(len(transformed)):
                for j in range(len(transformed[0])-1):
                    outfile.write(f"{transformed[i][j]},")
                outfile.write(f"{transformed[i][len(transformed[0])-1]}\n")

    def write_reconstructions(self, results_path, angles, reconstructed):
        os.mkdir(f"{results_path}/reconstructions")
        match self.filetypeGroup.checkedButton().text():
            case ".xy" | ".gr":
                for i in range(len(reconstructed)):
                    with open(f"{results_path}/reconstructions/scan_{i+1}{self.filetypeGroup.checkedButton().text()}", "w") as outfile:
                        for j in range(len(angles[i])):
                            outfile.write(f"{angles[i][j]}\t{reconstructed[i][j]}\n")
            case ".xye" | ".dat":
                for i in range(len(reconstructed)):
                        with open(f"{results_path}/reconstructions/scan_{i+1}{self.filetypeGroup.checkedButton().text()}", "w") as outfile:
                            for j in range(len(angles[i])):
                                outfile.write(f"{angles[i][j]}\t{reconstructed[i][j]}\t0\n")

    def export_results(self, angles, fitted, transformed, reconstructed, fig, errors=None):
        if not os.path.exists(f"{self.indirLabel.text()}/BaSSET_results"):
            os.mkdir(f"{self.indirLabel.text()}/BaSSET_results")

        export_time = datetime.now().strftime("%y%m%d-%H%M%S")
        results_path = f"{self.indirLabel.text()}/BaSSET_results/{export_time}_{self.algorithmGroup.checkedButton().text()}_{self.numComponentsSlider.value()}_{self.filetypeGroup.checkedButton().text()[1:]}"
        os.mkdir(results_path)

        fig.savefig(f"{results_path}/overview.jpg")
        self.write_summary(results_path, errors)
        self.write_components(results_path, angles, fitted)
        self.write_scores(results_path, transformed)
        self.write_reconstructions(results_path, angles, reconstructed)

class AboutDialog(qtw.QDialog):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("About BaSSET")
        self.setWindowIcon(qtg.QIcon(f"{parent.configpath}/assets/icon.png"))
        self.setWindowFlags(self.windowFlags() & ~qtc.Qt.WindowContextHelpButtonHint)

        layout = qtw.QVBoxLayout()

        self.program = qtw.QLabel("<h1>BaSSET</h1>")
        self.program.setAlignment(qtc.Qt.AlignCenter)
        layout.addWidget(self.program)

        self.logo = qtw.QLabel()
        self.logo.setPixmap(qtg.QPixmap(f"{parent.configpath}/assets/icon.png"))
        self.logo.setAlignment(qtc.Qt.AlignTop | qtc.Qt.AlignCenter)
        layout.addWidget(self.logo)

        """
        self.version = qtw.QLabel("Version: ALPHA_DEV")
        self.version.setAlignment(qtc.Qt.AlignCenter)
        layout.addWidget(self.version)
        """

        self.company = qtw.QLabel("Developed at NAFUMA Battery - University of Oslo")
        self.company.setAlignment(qtc.Qt.AlignCenter)
        self.company.resize(self.company.sizeHint())
        layout.addWidget(self.company)

        self.funding = qtw.QLabel("Funded by the Research Council of Norway<br>"
        "(<a href='https://prosjektbanken.forskningsradet.no/en/project/FORISS/325316'>BaSSET 325316</a>)")
        self.funding.setAlignment(qtc.Qt.AlignCenter)
        self.funding.setTextFormat(qtc.Qt.RichText)
        self.funding.setTextInteractionFlags(qtc.Qt.TextBrowserInteraction)
        self.funding.setOpenExternalLinks(True)
        layout.addWidget(self.funding)

        self.developer = qtw.QLabel("Developer: Eira T. North")
        self.developer.setAlignment(qtc.Qt.AlignCenter)
        layout.addWidget(self.developer)

        self.setLayout(layout)

        self.setFixedSize(qtc.QSize(280, 380))

def main():
    print("Starting BaSSET...")
    app = qtw.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()

if __name__=="__main__":
    main()