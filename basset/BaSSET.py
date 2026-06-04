"""
BaSSET GUI module
"""

import sys
import os
from datetime import datetime

from basset.utils import (
    analysis,
    file_worker,
    funcs
)

import numpy as np
import matplotlib.pyplot as plt
import PyQt6.QtCore as qtc
import PyQt6.QtWidgets as qtw
import PyQt6.QtGui as qtg


import platform
if platform.system() == "Windows":
    # Separates BaSSET from the "Pythonw.exe" ID so it can have its own tackbar icon
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("UiO.BaSSET")

plt.rcParams.update({
    #"font.family": "Arial",
    "font.size": 12, # Changes default font size for text, given in points (10 default)
    "figure.dpi": 100,
    "figure.figsize": (19.2, 10.8),
    "figure.constrained_layout.use": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "xtick.bottom": True,
    "xtick.minor.visible": True,
    "ytick.left": True,
    "ytick.right": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4, # major tick size in points (3.5 default)
    "xtick.minor.size": 2, # minor tick size in points (2 default)
    "ytick.major.size": 4, # major tick size in points (3.5 default)
    "ytick.minor.size": 2, # minor tick size in points (2 default)
    "legend.frameon": False,
    "legend.handlelength": 1,
    "savefig.dpi": 300, # figure dots per inch or 'figure'
    "savefig.bbox": "tight", # {tight, standard} (tight is incompatible with animations)
    "savefig.pad_inches": 0 # padding to be used, when bbox is set to 'tight'
})


class SciSpinBox(qtw.QDoubleSpinBox):
    """
    Modification of Qt's DoubleSpinBox, but using scientific notation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimum(-np.inf)
        self.setMaximum(np.inf)
        self.setDecimals(15)
        self.setMinimumWidth(110)
        self.multiplier = 10

    def textFromValue(self, value):
        """
        Converts float into two-decimal scientific notated text
        """
        return f"{value:.2e}"

    def valueFromText(self, text):
        """
        Converts text to float
        """
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text, pos):
        """
        Validates user input as acceptable
        """
        return qtg.QDoubleValidator.State.Acceptable, text, pos

    def stepBy(self, steps):
        """
        Changes arroe behavior to step exponent rather than mantissa
        """
        if steps > 0:
            self.setValue(self.value() * self.multiplier)
        else:
            self.setValue(self.value() / self.multiplier)

class MainWindow(qtw.QMainWindow):
    """
    Main BaSSET GUI window 
    """
    def __init__(self):
        super().__init__()

        self.angles = None
        self.intensities = None

        self.configpath = os.path.dirname(os.path.realpath(__file__))
        self.configfile = f"{self.configpath}/config.dat"

        self.setWindowIcon(qtg.QIcon(f"{self.configpath}/assets/icon.ico"))
        self.setWindowTitle("Battery Signal Selection and Enhancement Toolbox")
        self.resize(qtc.QSize(640,480))
        self.central_widget = qtw.QWidget(self)
        self.grid = qtw.QGridLayout()
        self.central_widget.setLayout(self.grid)
        self.setCentralWidget(self.central_widget)

        ##############################
        ##### Input data widgets #####
        ##############################
        self.input_format_layout = qtw.QGridLayout()
        self.grid.addLayout(self.input_format_layout, 0,0,2,1)

        self.input_format_title = qtw.QLabel("Input format")
        self.input_format_title.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.input_format_title.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.input_format_layout.addWidget(self.input_format_title, 0,0,1,3)

        self.input_format_group = qtw.QButtonGroup()
        self.theta_button = qtw.QRadioButton("2θ [°]")
        self.input_format_group.addButton(self.theta_button)
        self.input_format_layout.addWidget(self.theta_button, 1,0)

        self.q_button = qtw.QRadioButton("Q [Å⁻¹]")
        self.q_button.setChecked(True)
        self.input_format_group.addButton(self.q_button)
        self.input_format_layout.addWidget(self.q_button, 1,1)

        self.r_button = qtw.QRadioButton("r [Å]")
        self.input_format_group.addButton(self.r_button)
        self.input_format_layout.addWidget(self.r_button, 1,2)

        self.indir_options_layout = qtw.QHBoxLayout()
        self.input_format_layout.addLayout(self.indir_options_layout, 2,0,1,3)

        self.convert_to_q_checkbox = qtw.QCheckBox("Convert to Q")
        self.convert_to_q_checkbox.setToolTip("Uses supplied wavelength to display data in Q [Å⁻¹]")
        self.indir_options_layout.addWidget(self.convert_to_q_checkbox)
        self.input_format_group.buttonClicked.connect(
            lambda button: self.convert_to_q_checkbox.setEnabled(True)
            if button==self.theta_button
            else self.convert_to_q_checkbox.setDisabled(True)
        )
        self.r_button.clicked.connect(lambda: self.convert_to_q_checkbox.setChecked(False))

        self.wavelength_widget = qtw.QDoubleSpinBox(decimals=6)
        self.wavelength_widget.setMinimum(0)
        self.wavelength_widget.setSingleStep(0.000001)
        self.wavelength_widget.setSuffix(' Å')
        self.wavelength_widget.setValue(1.540598)
        self.wavelength_widget.setButtonSymbols(qtw.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.indir_options_layout.addWidget(self.wavelength_widget)
        self.input_format_group.buttonClicked.connect(
            lambda button: self.wavelength_widget.setEnabled(True)
            if button==self.theta_button
            else self.wavelength_widget.setDisabled(True)
        )
        self.input_format_group.buttonClicked.connect(
            lambda button: self.convert_to_q_checkbox.setChecked(False)
            if button!=self.theta_button
            else None
        )

        self.indir_layout = qtw.QHBoxLayout()
        self.grid.addLayout(self.indir_layout, 0,1,1,2)

        self.indir_label = qtw.QLabel("Select the folder containing your dataset")
        self.indir_label.setAlignment(qtc.Qt.AlignmentFlag.AlignRight)
        self.indir_label.setMinimumWidth(350)
        self.indir_label.setSizePolicy(qtw.QSizePolicy.Policy.Preferred, qtw.QSizePolicy.Policy.Fixed)
        self.indir_label.setFrameShape(qtw.QFrame.Shape.Panel)
        self.indir_label.setFrameShadow(qtw.QFrame.Shadow.Sunken)
        self.indir_layout.addWidget(self.indir_label)

        self.indir_button = qtw.QPushButton("Load dataset")
        self.indir_button.setSizePolicy(qtw.QSizePolicy.Policy.Fixed, qtw.QSizePolicy.Policy.Fixed)
        self.indir_layout.addWidget(self.indir_button)

        self.bkg_layout = qtw.QHBoxLayout()
        self.bkg_layout.setAlignment(qtc.Qt.AlignmentFlag.AlignRight)
        self.grid.addLayout(self.bkg_layout, 1,1,1,2)

        self.bkg_label = qtw.QLabel("Select a background for subtraction")
        self.bkg_label.setAlignment(qtc.Qt.AlignmentFlag.AlignRight)
        self.bkg_label.setMinimumWidth(350)
        self.bkg_label.setSizePolicy(qtw.QSizePolicy.Policy.Preferred, qtw.QSizePolicy.Policy.Fixed)
        self.bkg_label.setFrameShape(qtw.QFrame.Shape.Panel)
        self.bkg_label.setFrameShadow(qtw.QFrame.Shadow.Sunken)
        self.bkg_layout.addWidget(self.bkg_label)

        self.rm_bkg_button = qtw.QPushButton("Remove")
        self.rm_bkg_button.setSizePolicy(qtw.QSizePolicy.Policy.Fixed, qtw.QSizePolicy.Policy.Fixed)
        self.bkg_layout.addWidget(self.rm_bkg_button)

        self.bkg_scale_spinbox = SciSpinBox()
        self.bkg_scale_spinbox.setMinimum(0)
        self.bkg_scale_spinbox.setValue(1)
        self.bkg_scale_spinbox.setToolTip('Scale your background to match dataset\n' \
                                          '(View in "Plot input dataset")')
        self.bkg_layout.addWidget(self.bkg_scale_spinbox)

        self.getbkg_button = qtw.QPushButton("Load background")
        self.getbkg_button.setSizePolicy(qtw.QSizePolicy.Policy.Fixed, qtw.QSizePolicy.Policy.Fixed)
        self.bkg_layout.addWidget(self.getbkg_button)

        self.plot_data_button = qtw.QPushButton("Plot input dataset")
        self.plot_data_button.setSizePolicy(qtw.QSizePolicy.Policy.MinimumExpanding, qtw.QSizePolicy.Policy.Ignored)
        self.grid.addWidget(self.plot_data_button, 2,0)

        self.xrange_layout = qtw.QGridLayout()
        self.grid.addLayout(self.xrange_layout, 3,0)

        self.plot_xrange_label = qtw.QLabel("Analyze x-axis range")
        self.plot_xrange_label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.plot_xrange_label.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.xrange_layout.addWidget(self.plot_xrange_label, 2,0,1,2)

        self.plot_xmin_spinbox = qtw.QDoubleSpinBox()
        self.plot_xmin_spinbox.setMinimum(0)
        self.plot_xmin_spinbox.setMaximum(999.99)
        self.plot_xmin_spinbox.setValue(0)
        self.plot_xmin_spinbox.setDecimals(2)
        self.plot_xmin_spinbox.setSingleStep(0.01)
        self.xrange_layout.addWidget(self.plot_xmin_spinbox, 3,0)
        self.plot_xmin_spinbox.setToolTip("Minimum x-value in plots\n" \
                                           "(if min equals max, use full dataset)")

        self.plot_xmax_spinbox = qtw.QDoubleSpinBox()
        self.plot_xmax_spinbox.setMinimum(0)
        self.plot_xmax_spinbox.setMaximum(999.99)
        self.plot_xmax_spinbox.setValue(0)
        self.plot_xmax_spinbox.setDecimals(2)
        self.plot_xmax_spinbox.setSingleStep(0.01)
        self.xrange_layout.addWidget(self.plot_xmax_spinbox, 3,1)
        self.plot_xmax_spinbox.setToolTip("Maximum x-value in plots\n" \
                                           "(if min equals max, use full dataset)")

        #############################
        ##### Algorithm widgets #####
        #############################

        self.algorithm_layout = qtw.QGridLayout()
        self.grid.addLayout(self.algorithm_layout, 2,1)

        self.algorithm_group = qtw.QButtonGroup()
        self.algorithm_title = qtw.QLabel("Algorithm")
        self.algorithm_title.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.algorithm_title.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.algorithm_layout.addWidget(self.algorithm_title, 0,0,1,3)

        self.PCA_button = qtw.QRadioButton("PCA")
        self.PCA_button.setChecked(True)
        self.PCA_button.setToolTip("Principal Component Analysis (scikit-learn)")
        self.algorithm_layout.addWidget(self.PCA_button, 1,0)

        self.NMF_button = qtw.QRadioButton("NMF")
        self.NMF_button.setToolTip("Non-Negative Matrix Factorization (scikit-learn)")
        self.algorithm_layout.addWidget(self.NMF_button, 1,1)

        self.ICA_button = qtw.QRadioButton("ICA")
        self.ICA_button.setToolTip("Independent Component Analysis (scikit-learn)")
        self.algorithm_layout.addWidget(self.ICA_button, 1,2)

        self.SNMF_button = qtw.QRadioButton("SNMF")
        self.SNMF_button.setToolTip("Stretched Non-Negative Matrix Factorization (diffpy)\n" \
                                   "WARNING: The SNMF algorithm takes a lot of time.\n" \
                                   "Error calculation will not be performed.")
        self.algorithm_layout.addWidget(self.SNMF_button, 2,0)

        self.calc_err_checkbox = qtw.QCheckBox("Calculate errors")
        self.calc_err_checkbox.setChecked(True)
        self.calc_err_checkbox.setToolTip("Calculates errors for 1 to 10 components")
        self.algorithm_layout.addWidget(self.calc_err_checkbox, 2,1)
        self.algorithm_group.buttonClicked.connect(
            lambda button: self.calc_err_checkbox.setDisabled(True)
            if button==self.SNMF_button
            else self.calc_err_checkbox.setEnabled(True)
        )
        self.algorithm_group.buttonClicked.connect(
            lambda button: self.calc_err_checkbox.setChecked(False)
            if button==self.SNMF_button
            else None
        )

        self.algorithm_group.addButton(self.PCA_button)
        self.algorithm_group.addButton(self.NMF_button)
        self.algorithm_group.addButton(self.ICA_button)
        self.algorithm_group.addButton(self.SNMF_button)

        self.comp_num_layout = qtw.QGridLayout()
        self.grid.addLayout(self.comp_num_layout, 3,1)

        self.comp_num_title = qtw.QLabel("Number of Components")
        self.comp_num_title.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.comp_num_title.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.comp_num_layout.addWidget(self.comp_num_title, 0,0)

        self.num_components_slider_layout = qtw.QHBoxLayout()
        self.comp_num_layout.addLayout(self.num_components_slider_layout, 1,0)

        self.comp_num_slider = qtw.QSlider(qtc.Qt.Orientation.Horizontal)
        self.comp_num_slider.setTickPosition(qtw.QSlider.TickPosition.TicksBothSides)
        self.comp_num_slider.setTickInterval(1)
        self.comp_num_slider.setMinimum(2)
        self.comp_num_slider.setMaximum(10)
        self.comp_num_slider.setValue(4)
        self.comp_num_slider.setSingleStep(1)
        self.comp_num_slider.setPageStep(1)
        self.comp_num_slider.setSizePolicy(qtw.QSizePolicy.Policy.Minimum,qtw.QSizePolicy.Policy.Fixed)
        self.num_components_slider_layout.addWidget(self.comp_num_slider)

        self.comp_num_label = qtw.QLabel(str(self.comp_num_slider.value()))
        self.comp_num_label.setSizePolicy(qtw.QSizePolicy.Policy.Fixed, qtw.QSizePolicy.Policy.Fixed)
        self.num_components_slider_layout.addWidget(self.comp_num_label)


        self.algorithm_parameters_layout = qtw.QGridLayout()
        self.grid.addLayout(self.algorithm_parameters_layout, 4,1)

        self.algorithm_parameters_title = qtw.QLabel("Algorithm Parameters")
        self.algorithm_parameters_title.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.algorithm_parameters_title.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.algorithm_parameters_layout.addWidget(self.algorithm_parameters_title, 0,0, 1,3)

        ##########################
        ##### PCA parameters #####
        ##########################
        self.PCA_whiten_checkbox = qtw.QCheckBox("Whiten")
        self.algorithm_parameters_layout.addWidget(self.PCA_whiten_checkbox, 1,0)
        self.PCA_whiten_checkbox.setToolTip("[whiten]\n" \
        "Whitening will remove some information from the transformed signal\n" \
        "(the relative variance scales of the components) but can sometime\n" \
        "improve the predictive accuracy of the downstream estimators\n" \
        "by making their data respect some hard-wired assumptions")

        self.PCA_solver_dropdown = qtw.QComboBox()
        self.PCA_solver_dropdown.addItems(['auto', 'full', 'covariance_eigh', 'arpack', 'randomized'])
        self.algorithm_parameters_layout.addWidget(self.PCA_solver_dropdown, 1,1)
        self.PCA_solver_dropdown.setToolTip("[svd_solver]\n" \
        "The type of Singular Value Decomposition solver to use:\n" \
        "auto (default): Chooses solver based on size of dataset and number of components\n" \
        "full: Runs exact full SVD\n" \
        "covariance_eigh: Precomputes covarience for eigenvalue decompositon.\n" \
        "    Efficient for many scans of few datapoints (rare for scattering)\n" \
        "arpack: Runs SVD truncated to number of components.\n" \
        "    Requires fewer components than number of scans\n" \
        "randomized: Runs randomized SVD")

        # Only for 'arpack' solver
        self.PCA_tol_spinbox = SciSpinBox()
        self.PCA_tol_spinbox.setMinimum(0)
        self.PCA_tol_spinbox.setValue(0)
        self.algorithm_parameters_layout.addWidget(self.PCA_tol_spinbox, 1,2)
        self.PCA_tol_spinbox.setToolTip("[tol]\n" \
        "Tolerance for singular values using 'arpack'")
        self.PCA_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.PCA_tol_spinbox.show()
            if currentText=='arpack'
            else self.PCA_tol_spinbox.hide()
        )

        # Only for 'randomized' solver
        self.PCA_iterated_power_spinbox = qtw.QSpinBox()
        self.PCA_iterated_power_spinbox.setMinimum(0)
        self.PCA_iterated_power_spinbox.setMaximum(999999)
        self.PCA_iterated_power_spinbox.setValue(0)
        self.PCA_iterated_power_spinbox.setSingleStep(100)
        self.algorithm_parameters_layout.addWidget(self.PCA_iterated_power_spinbox, 2,0)
        self.PCA_iterated_power_spinbox.setToolTip("[iterated_power]\n" \
        "Number of iterations for the power method in 'randomized'")
        self.PCA_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.PCA_iterated_power_spinbox.show()
            if currentText=='randomized'
            else self.PCA_iterated_power_spinbox.hide()
        )

        # Only for 'randomized' solver
        self.PCA_iterated_power_auto_checkbox = qtw.QCheckBox("auto")
        self.algorithm_parameters_layout.addWidget(self.PCA_iterated_power_auto_checkbox, 2,1)
        self.PCA_iterated_power_auto_checkbox.setToolTip("[iterated_power]\n" \
        "Automatically choose number of iterations")
        self.PCA_iterated_power_auto_checkbox.clicked.connect(
            lambda state: self.PCA_iterated_power_spinbox.setDisabled(True)
            if state
            else self.PCA_iterated_power_spinbox.setDisabled(False)
        )
        self.PCA_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.PCA_iterated_power_auto_checkbox.show()
            if currentText=='randomized'
            else self.PCA_iterated_power_auto_checkbox.hide()
        )

        # Only for 'randomized' solver
        self.PCA_n_oversampled_spinbox = qtw.QSpinBox()
        self.PCA_n_oversampled_spinbox.setMinimum(0)
        self.PCA_n_oversampled_spinbox.setMaximum(50)
        self.PCA_n_oversampled_spinbox.setValue(10)
        self.PCA_n_oversampled_spinbox.setSingleStep(1)
        self.algorithm_parameters_layout.addWidget(self.PCA_n_oversampled_spinbox, 2,2)
        self.PCA_n_oversampled_spinbox.setToolTip("[n_oversamples]\n" \
        "Additional number of random vectors to sample using 'randomized'")
        self.PCA_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.PCA_n_oversampled_spinbox.show()
            if currentText=='randomized'
            else self.PCA_n_oversampled_spinbox.hide()
        )

        # Only for 'randomized' solver // Not for 'arpack' solver
        self.PCA_power_iteration_normalizer_dropdown = qtw.QComboBox()
        self.PCA_power_iteration_normalizer_dropdown.addItems(['auto', 'QR', 'LU', 'none'])
        self.algorithm_parameters_layout.addWidget(self.PCA_power_iteration_normalizer_dropdown, 1,2)
        self.PCA_power_iteration_normalizer_dropdown.setToolTip("[power_iteration_normalizer]\n" \
        "Power iteration normalizer using 'randomized'")
        self.PCA_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.PCA_power_iteration_normalizer_dropdown.show()
            if currentText=='randomized'
            else self.PCA_power_iteration_normalizer_dropdown.hide()
        )

        # Ignored PCA parameters:
        # random_state

        self.PCA_algorithm_widgets = [self.PCA_whiten_checkbox, self.PCA_solver_dropdown,
                                    self.PCA_tol_spinbox, self.PCA_iterated_power_spinbox,
                                    self.PCA_iterated_power_auto_checkbox, self.PCA_n_oversampled_spinbox,
                                    self.PCA_n_oversampled_spinbox, self.PCA_power_iteration_normalizer_dropdown]

        ##########################
        ##### NMF parameters #####
        ##########################
        self.NMF_init_dropdown = qtw.QComboBox()
        self.NMF_init_dropdown.addItems(["nndsvda", "random", "nndsvd", "nndsvdar"])
        self.algorithm_parameters_layout.addWidget(self.NMF_init_dropdown, 1,0)
        self.NMF_init_dropdown.setToolTip("[init]\n" \
        "Method used to initialize the procedure:\n" \
        "('nndsvda' is recommended for PDF and 'nndsvd' is recommended for XRD)\n" \
        "nndsvda (default): Better when sparsity is not desired (PDF)\n" \
        "random: Random non-negative matrices\n" \
        "nndsvd: Non-negative Double Singular Value Decomposition (better for sparseness) (XRD)\n" \
        "nndsvdar: Faster, less accurate alternative to NNDSVDa when sparsity is not desired")

        self.NMF_solver_dropdown = qtw.QComboBox()
        self.NMF_solver_dropdown.addItems(["cd", "mu"])
        self.algorithm_parameters_layout.addWidget(self.NMF_solver_dropdown, 1,1)
        self.NMF_solver_dropdown.setToolTip("[solver]\n" \
        "Numerical solver to use:\n" \
        "cd (default): Coordinate Descent\n" \
        "mu: Multiplicative Update\n" \
        "('mu' gives poor results with 'nndsvd' as it cannot update zeros in initialization)")

        self.NMF_max_iter_spinbox = qtw.QSpinBox()
        self.NMF_max_iter_spinbox.setMinimum(0)
        self.NMF_max_iter_spinbox.setMaximum(999999)
        self.NMF_max_iter_spinbox.setValue(2500)
        self.algorithm_parameters_layout.addWidget(self.NMF_max_iter_spinbox, 2,0)
        self.NMF_max_iter_spinbox.setToolTip("[max_iter]\n" \
        "Maximum number of iterations before timing out")

        self.NMF_tol_spinbox = SciSpinBox()
        self.NMF_tol_spinbox.setMinimum(0)
        self.NMF_tol_spinbox.setValue(1e-4)
        self.algorithm_parameters_layout.addWidget(self.NMF_tol_spinbox, 2,1)
        self.NMF_tol_spinbox.setToolTip("[tol]\n" \
        "Tolerance of the stopping condition")

        # Used in 'cd' solver
        self.NMF_l1_ratio_spinbox = qtw.QDoubleSpinBox()
        self.NMF_l1_ratio_spinbox.setMinimum(0)
        self.NMF_l1_ratio_spinbox.setMaximum(1)
        self.NMF_l1_ratio_spinbox.setValue(0)
        self.NMF_l1_ratio_spinbox.setSingleStep(0.05)
        self.algorithm_parameters_layout.addWidget(self.NMF_l1_ratio_spinbox, 1,2)
        self.NMF_l1_ratio_spinbox.setToolTip("[l1_ratio]\n" \
        "Regularization mixing parameter:\n" \
        "(0, default): elementwise L2 penalty aka. Frobenius Norm\n"
        "(1): elementwwise L1 penalty (better for sparseness)\n")
        self.NMF_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.NMF_l1_ratio_spinbox.show()
            if currentText=='cd'
            else self.NMF_l1_ratio_spinbox.hide()
        )

        # Only for 'mu' solver
        self.NMF_beta_loss_dropdown = qtw.QComboBox()
        self.NMF_beta_loss_dropdown.addItems(["frobenius", "kullback-leibler"]) # "itakura-saito" not included as it cannot have zeros in input data, which XRD/PDF often can have
        self.algorithm_parameters_layout.addWidget(self.NMF_beta_loss_dropdown, 1,2)
        self.NMF_beta_loss_dropdown.setToolTip("[beta_loss]\n" \
        "Beta divergence to be minimized," \
        "measuring the distance between X and the dot product WH using 'mu':\n" \
        "frobenius is default")
        self.NMF_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.NMF_beta_loss_dropdown.show()
            if currentText=='mu'
            else self.NMF_beta_loss_dropdown.hide()
        )

        self.NMF_alpha_W_spinbox = qtw.QDoubleSpinBox()
        self.NMF_alpha_W_spinbox.setMinimum(0)
        self.NMF_alpha_W_spinbox.setMaximum(10)
        self.NMF_alpha_W_spinbox.setValue(0)
        self.NMF_alpha_W_spinbox.setDecimals(5)
        self.NMF_alpha_W_spinbox.setSingleStep(0.00005)
        self.algorithm_parameters_layout.addWidget(self.NMF_alpha_W_spinbox, 3,0)
        self.NMF_alpha_W_spinbox.setToolTip("[alpha_W]\n" \
        "Constant that multiplies the regularization terms of the mixing of features:\n" \
        "(Regularization is a penalty term to constrain parameter complexity and reduce overfitting)\n" \
        "0 (default) means no regularization")

        self.NMF_alpha_H_spinbox = qtw.QDoubleSpinBox()
        self.NMF_alpha_H_spinbox.setMinimum(0)
        self.NMF_alpha_H_spinbox.setMaximum(10)
        self.NMF_alpha_H_spinbox.setValue(0)
        self.NMF_alpha_H_spinbox.setDecimals(5)
        self.NMF_alpha_H_spinbox.setSingleStep(0.00005)
        self.NMF_alpha_H_spinbox.setDisabled(True)
        self.algorithm_parameters_layout.addWidget(self.NMF_alpha_H_spinbox, 3,1)
        self.NMF_alpha_H_spinbox.setToolTip("[alpha_H]\n" \
        "Constant that multiplies the regularization terms of the features:\n" \
        "0 means no regularization. By default same as alpha_W ")

        self.NMF_alpha_H_same_checkbox = qtw.QCheckBox("same")
        self.NMF_alpha_H_same_checkbox.setChecked(True)
        self.algorithm_parameters_layout.addWidget(self.NMF_alpha_H_same_checkbox, 3,2)
        self.NMF_alpha_H_same_checkbox.setToolTip("[alpha_H]\n" \
        "Use same regularization constant for features and mixing")
        self.NMF_alpha_H_same_checkbox.clicked.connect(
            lambda state: self.NMF_alpha_H_spinbox.setDisabled(True)
            if state
            else self.NMF_alpha_H_spinbox.setDisabled(False)
        )
        self.NMF_alpha_H_same_checkbox.clicked.connect(
            lambda state: self.NMF_alpha_H_spinbox.setValue(self.NMF_alpha_W_spinbox.value())
            if state
            else None
        )
        self.NMF_alpha_W_spinbox.valueChanged.connect(
            lambda value: self.NMF_alpha_H_spinbox.setValue(value)
            if self.NMF_alpha_H_same_checkbox.isChecked()
            else None
        )

        self.NMF_rescale_checkbox = qtw.QCheckBox("Rescale")
        self.algorithm_parameters_layout.addWidget(self.NMF_rescale_checkbox, 2,2)
        self.NMF_rescale_checkbox.setToolTip("Rescales scores to sum to 1")


        # Ignored NMF parmeters:
        # random_state
        # verbose
        # shuffle

        self.NMF_algorithm_widgets = [self.NMF_init_dropdown, self.NMF_solver_dropdown,
                                    self.NMF_max_iter_spinbox, self.NMF_tol_spinbox,
                                    self.NMF_l1_ratio_spinbox, self.NMF_beta_loss_dropdown,
                                    self.NMF_alpha_W_spinbox, self.NMF_alpha_H_spinbox,
                                    self.NMF_alpha_H_same_checkbox, self.NMF_rescale_checkbox]

        ##########################
        ##### ICA parameters #####
        ##########################
        self.ICA_algorithm_dropdown = qtw.QComboBox()
        self.ICA_algorithm_dropdown.addItems(['parallel', 'deflation'])
        self.algorithm_parameters_layout.addWidget(self.ICA_algorithm_dropdown, 1,0)
        self.ICA_algorithm_dropdown.setToolTip("[algorithm]\n" \
        "Specify which algorithm to use:\n" \
        "parallel is default")

        self.ICA_whiten_dropdown = qtw.QComboBox()
        self.ICA_whiten_dropdown.addItems(['unit-variance', 'arbitrary-variance', 'False'])
        self.algorithm_parameters_layout.addWidget(self.ICA_whiten_dropdown, 1,1)
        self.ICA_whiten_dropdown.setToolTip("[whiten]\n" \
        "Whitening strategy to use. False means no whitening:\n" \
        "unit-variance (default): the whitening matrix is rescaled to ensure each recovered source has unit variance\n" \
        "arbitrary-variance: a whitening with variance arbitrary is used\n" \
        "False: Data is considered whitened and no whitening is performed")

        self.ICA_max_iter_spinbox = qtw.QSpinBox()
        self.ICA_max_iter_spinbox.setMinimum(0)
        self.ICA_max_iter_spinbox.setMaximum(999999)
        self.ICA_max_iter_spinbox.setValue(2500)
        self.algorithm_parameters_layout.addWidget(self.ICA_max_iter_spinbox, 2,0)
        self.ICA_max_iter_spinbox.setToolTip("[max_iter]\n" \
        "Maximum number of iterations during fit")

        self.ICA_tol_spinbox = SciSpinBox()
        self.ICA_tol_spinbox.setMinimum(0)
        self.ICA_tol_spinbox.setValue(1e-4)
        self.algorithm_parameters_layout.addWidget(self.ICA_tol_spinbox, 2,1)
        self.ICA_tol_spinbox.setToolTip("[tol]\n" \
        "A positive scalar giving the tolerance at which the un-mixing matrix is considered to have converged")

        self.ICA_fun_dropdown = qtw.QComboBox()
        self.ICA_fun_dropdown.addItems(['logcosh', 'exp', 'cube'])
        self.algorithm_parameters_layout.addWidget(self.ICA_fun_dropdown, 1,2)
        self.ICA_fun_dropdown.setToolTip("[fun]\n" \
        "The functional form of the G function used in the approximation to neg-entropy:\n" \
        "logcosh is default")

        self.ICA_whiten_solver_dropdown = qtw.QComboBox()
        self.ICA_whiten_solver_dropdown.addItems(['svd', 'eigh'])
        self.algorithm_parameters_layout.addWidget(self.ICA_whiten_solver_dropdown, 2,2)
        self.ICA_whiten_solver_dropdown.setToolTip("[whiten_solver]\n" \
        "The solver to use for whitening:\n" \
        "svd (default): More numerically stable if the problem is degenerate, and often faster for scattering\n" \
        "eigh: More memory efficient when there are more scans than datapoints (rare in scattering)")

        # Ignored ICA parameters:
        # fun_args
        # w_init
        # random_state

        self.ICA_algorithm_widgets = [self.ICA_algorithm_dropdown, self.ICA_whiten_dropdown,
                                    self.ICA_max_iter_spinbox, self.ICA_tol_spinbox,
                                    self.ICA_fun_dropdown, self.ICA_whiten_solver_dropdown]

        ###########################
        ##### SNMF parameters #####
        ###########################
        self.SNMF_min_iter_spinbox = qtw.QSpinBox()
        self.SNMF_min_iter_spinbox.setMinimum(0)
        self.SNMF_min_iter_spinbox.setMaximum(999999)
        self.SNMF_min_iter_spinbox.setValue(20)
        self.algorithm_parameters_layout.addWidget(self.SNMF_min_iter_spinbox, 2,0)
        self.SNMF_min_iter_spinbox.setToolTip("[min_iter]\n" \
        "Minimum number of iterations before terminating optimzation")

        self.SNMF_max_iter_spinbox = qtw.QSpinBox()
        self.SNMF_max_iter_spinbox.setMinimum(0)
        self.SNMF_max_iter_spinbox.setMaximum(999999)
        self.SNMF_max_iter_spinbox.setValue(500)
        self.algorithm_parameters_layout.addWidget(self.SNMF_max_iter_spinbox, 2,1)
        self.SNMF_max_iter_spinbox.setToolTip("[max_iter]\n" \
        "Maximum number of iterations before terminating optimzation")

        self.SNMF_tol_spinbox = SciSpinBox()
        self.SNMF_tol_spinbox.setMinimum(0)
        self.SNMF_tol_spinbox.setValue(5e-07)
        self.algorithm_parameters_layout.addWidget(self.SNMF_tol_spinbox, 2,2)
        self.SNMF_tol_spinbox.setToolTip("[tol]\n" \
        "Convergence threshold.\n" \
        "Minimum fractional improvment to allow terminating optimization")

        self.SNMF_rho_spinbox = SciSpinBox()
        self.SNMF_rho_spinbox.setMinimum(0)
        self.SNMF_rho_spinbox.setValue(0)
        self.SNMF_rho_spinbox.setDecimals(0)
        #self.SNMFrho_Spinbox.setMaximum(10000000000)
        #self.SNMFrho_Spinbox.setSingleStep(1)
        self.algorithm_parameters_layout.addWidget(self.SNMF_rho_spinbox, 3,0)
        self.SNMF_rho_spinbox.setToolTip("[rho]\n"
        "Stretching factor // Stretching regularization hyperparameter. Controls stretching pentalty. \n" \
        "Typically adjusted in powers of 10.\n" \
        "Zero (default) corresponds to no stretching")

        self.SNMF_eta_spinbox = qtw.QDoubleSpinBox()
        self.SNMF_eta_spinbox.setMinimum(0)
        self.SNMF_eta_spinbox.setValue(0)
        self.SNMF_eta_spinbox.setDecimals(5)
        #self.SNMFeta_DSpinbox.setMaximum(10)
        #self.SNMFeta_DSpinbox.setSingleStep(0.00005)
        self.algorithm_parameters_layout.addWidget(self.SNMF_eta_spinbox, 3,1)
        self.SNMF_eta_spinbox.setToolTip("[eta]\n" \
        "Sparsity factor. Should be set to zero (default) for non-sparse data such as PDF.\n" \
        "Can be used to improve results for sparse data such as XRD,\nbut due to instability should be used only faster selecting the best value for rho.\n" \
        "Suggested adjustment is by powers of 2")

        # Ignored SNMF parmeters:
        # random_state
        # verbose
        # show_plots

        self.SNMF_algorithm_widgets = [self.SNMF_min_iter_spinbox, self.SNMF_max_iter_spinbox,
                                     self.SNMF_tol_spinbox, self.SNMF_rho_spinbox,
                                     self.SNMF_eta_spinbox]

        #####################################
        ##### Analysis and plot widgets #####
        #####################################

        self.run_analysis_button = qtw.QPushButton("Run analysis")
        self.run_analysis_button.setSizePolicy(qtw.QSizePolicy.Policy.MinimumExpanding, qtw.QSizePolicy.Policy.Preferred)
        self.grid.addWidget(self.run_analysis_button, 2,2)

        self.results_layout = qtw.QGridLayout()
        self.grid.addLayout(self.results_layout, 3,2)

        self.export_results_checkbox = qtw.QCheckBox("Export results")
        self.export_results_checkbox.setToolTip("Exports plot of dataset, and analysis plot and results\n" \
        "For the sake of analysis in other programs data is exported in input format (2θ not converted to Q)")
        self.results_layout.addWidget(self.export_results_checkbox, 0,0)

        self.export_recons_checkbox = qtw.QCheckBox("Reconstructions")
        self.export_recons_checkbox.setToolTip("Exports the reconstructed dataset – scan by scan")
        self.export_recons_checkbox.setDisabled(True)
        self.results_layout.addWidget(self.export_recons_checkbox, 0,1)
        self.export_results_checkbox.clicked.connect(
            lambda state: self.export_recons_checkbox.setDisabled(False)
            if state
            else self.export_recons_checkbox.setDisabled(True)
        )
        self.export_results_checkbox.clicked.connect(
            lambda state: self.export_recons_checkbox.setChecked(False)
            if not state
            else None
        )

        self.export_diffs_checkbox = qtw.QCheckBox("Differences")
        self.export_diffs_checkbox.setToolTip("Exports the difference between dataset and reconstruction – scan by scan")
        self.export_diffs_checkbox.setDisabled(True)
        self.results_layout.addWidget(self.export_diffs_checkbox, 1,1)
        self.export_results_checkbox.clicked.connect(
            lambda state: self.export_diffs_checkbox.setDisabled(False)
            if state
            else self.export_diffs_checkbox.setDisabled(True)
        )
        self.export_results_checkbox.clicked.connect(
            lambda state: self.export_diffs_checkbox.setChecked(False)
            if not state
            else None
        )

        self.export_comp_contribute_checkbox = qtw.QCheckBox("Component contributions")
        self.export_comp_contribute_checkbox.setToolTip("Exports the partial component-wise reconstruction\n" \
                                                       "Each component multiplied by its own scoring – scan by scan")
        self.export_comp_contribute_checkbox.setDisabled(True)
        self.results_layout.addWidget(self.export_comp_contribute_checkbox, 1,0)
        self.export_results_checkbox.clicked.connect(
            lambda state: self.export_comp_contribute_checkbox.setDisabled(False)
            if state
            else self.export_comp_contribute_checkbox.setDisabled(True)
        )
        self.export_results_checkbox.clicked.connect(
            lambda state: self.export_comp_contribute_checkbox.setChecked(False)
            if not state
            else None
        )


        self.recon_plot_layout = qtw.QGridLayout()
        self.grid.addLayout(self.recon_plot_layout, 4,2)
        self.recon_plot_layout.setAlignment(qtc.Qt.AlignmentFlag.AlignTop)

        self.reconstruct_label = qtw.QLabel("Display Reconstruction Scan #")
        self.reconstruct_label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.reconstruct_label.setSizePolicy(qtw.QSizePolicy.Policy.Minimum, qtw.QSizePolicy.Policy.Fixed)
        self.recon_plot_layout.addWidget(self.reconstruct_label, 4,0,1,2)

        self.reconstruct_widgets = []

        self.reconstruct_widget0 = qtw.QSpinBox()
        self.reconstruct_widget0.setMinimum(0)
        self.reconstruct_widget0.setMaximum(9999)
        self.recon_plot_layout.addWidget(self.reconstruct_widget0, 5,0)
        self.reconstruct_widget0.setToolTip("If any are '0', will reconstruct uniform distribution of scans")
        self.reconstruct_widgets.append(self.reconstruct_widget0)

        self.reconstruct_widget1 = qtw.QSpinBox()
        self.reconstruct_widget1.setMinimum(0)
        self.reconstruct_widget1.setMaximum(9999)
        self.recon_plot_layout.addWidget(self.reconstruct_widget1, 5,1)
        self.reconstruct_widgets.append(self.reconstruct_widget1)

        self.reconstruct_widget2 = qtw.QSpinBox()
        self.reconstruct_widget2.setMinimum(0)
        self.reconstruct_widget2.setMaximum(9999)
        self.recon_plot_layout.addWidget(self.reconstruct_widget2, 6,0)
        self.reconstruct_widgets.append(self.reconstruct_widget2)

        self.reconstruct_widget3 = qtw.QSpinBox()
        self.reconstruct_widget3.setMinimum(0)
        self.reconstruct_widget3.setMaximum(9999)
        self.recon_plot_layout.addWidget(self.reconstruct_widget3, 6,1)
        self.reconstruct_widgets.append(self.reconstruct_widget3)

        self.reconstruct_widget4 = qtw.QSpinBox()
        self.reconstruct_widget4.setMinimum(0)
        self.reconstruct_widget4.setMaximum(9999)
        self.recon_plot_layout.addWidget(self.reconstruct_widget4, 7,0)
        self.reconstruct_widgets.append(self.reconstruct_widget4)

        self.reconstruct_widget5 = qtw.QSpinBox()
        self.reconstruct_widget5.setMinimum(0)
        self.reconstruct_widget5.setMaximum(9999)
        self.recon_plot_layout.addWidget(self.reconstruct_widget5, 7,1)
        self.reconstruct_widgets.append(self.reconstruct_widget5)

        self.reconstruct_widget6 = qtw.QSpinBox()
        self.reconstruct_widget6.setMinimum(0)
        self.reconstruct_widget6.setMaximum(9999)
        self.recon_plot_layout.addWidget(self.reconstruct_widget6, 8,0)
        self.reconstruct_widgets.append(self.reconstruct_widget6)

        self.reconstruct_widget7 = qtw.QSpinBox()
        self.reconstruct_widget7.setMinimum(0)
        self.reconstruct_widget7.setMaximum(9999)
        self.recon_plot_layout.addWidget(self.reconstruct_widget7, 8,1)
        self.reconstruct_widgets.append(self.reconstruct_widget7)

        self.reconstruct_widget8 = qtw.QSpinBox()
        self.reconstruct_widget8.setMinimum(0)
        self.reconstruct_widget8.setMaximum(9999)
        self.recon_plot_layout.addWidget(self.reconstruct_widget8, 9,0)
        self.reconstruct_widgets.append(self.reconstruct_widget8)

        self.reconstruct_widget9 = qtw.QSpinBox()
        self.reconstruct_widget9.setMinimum(0)
        self.reconstruct_widget9.setMaximum(9999)
        self.recon_plot_layout.addWidget(self.reconstruct_widget9, 9,1)
        self.reconstruct_widgets.append(self.reconstruct_widget9)

        ########################
        ##### Top bar menu #####
        ########################
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        open_indir_button = qtg.QAction("Open file directory", self)
        open_indir_button.triggered.connect(
            lambda: qtg.QDesktopServices.openUrl(qtc.QUrl.fromLocalFile(self.indir_label.text()))
        )
        file_menu.addAction(open_indir_button)

        config_button = qtg.QAction("Open config file", self)
        config_button.triggered.connect(
            lambda: qtg.QDesktopServices.openUrl(qtc.QUrl.fromLocalFile(f"{self.configpath}/config.dat"))
        )
        file_menu.addAction(config_button)

        file_menu.addSeparator()

        self.file_submenu_recent_dirs = file_menu.addMenu("Recent datasets")
        self.file_submenu_recent_dirs_separator = self.file_submenu_recent_dirs.addSeparator()

        self.clear_recent_dirs_button = qtg.QAction("Clear recently opened", self)
        self.clear_recent_dirs_button.triggered.connect(
            lambda: [self.file_submenu_recent_dirs.removeAction(action) for action in self.file_submenu_recent_dirs.actions()[:-2]]
        )
        self.clear_recent_dirs_button.triggered.connect(self.update_config_file)
        self.file_submenu_recent_dirs.addAction(self.clear_recent_dirs_button)

        self.file_submenu_recent_bkgs = file_menu.addMenu("Recent backgrounds")
        self.file_submenu_recent_bkgs_separator = self.file_submenu_recent_bkgs.addSeparator()

        self.clear_recent_bkgs_button = qtg.QAction("Clear recently opened", self)
        self.clear_recent_bkgs_button.triggered.connect(
            lambda: [self.file_submenu_recent_bkgs.removeAction(action) for action in self.file_submenu_recent_bkgs.actions()[:-2]]
        )
        self.clear_recent_bkgs_button.triggered.connect(self.update_config_file)
        self.file_submenu_recent_bkgs.addAction(self.clear_recent_bkgs_button)

        file_menu.addSeparator()

        exit_button = qtg.QAction("Exit", self)
        exit_button.triggered.connect(self.closeEvent)
        file_menu.addAction(exit_button)


        help_menu = menu.addMenu("Help")
        github_button = qtg.QAction("Source code", self)
        github_button.triggered.connect(
            lambda: qtg.QDesktopServices.openUrl(qtc.QUrl("https://github.com/etnorth/BaSSET"))
        )
        help_menu.addAction(github_button)

        self.about_window = None
        about_button = qtg.QAction(qtg.QIcon(f"{self.configpath}/assets/icon.png"), "About", self)
        about_button.triggered.connect(self.about_button_clicked)
        help_menu.addAction(about_button)

        ##### Setup GUI according to config
        if os.path.exists(self.configfile):
            self.read_config_file()

        self.display_algorithm_widgets()
        self.display_reconstruction_widgets()

        ################################
        ##### Function connections #####
        ################################
        self.indir_button.clicked.connect(self.set_indir)
        self.indir_button.clicked.connect(self.update_config_file)
        self.getbkg_button.clicked.connect(self.set_bkgfile)
        self.getbkg_button.clicked.connect(self.update_config_file)
        self.bkg_scale_spinbox.valueChanged.connect(self.update_config_file)
        self.rm_bkg_button.clicked.connect(
            lambda: self.bkg_label.setText("Select a background for subtraction")
        )
        self.rm_bkg_button.clicked.connect(self.update_config_file)
        self.input_format_group.buttonClicked.connect(self.update_config_file)
        self.input_format_group.buttonClicked.connect(self.set_widget_limits)
        self.convert_to_q_checkbox.clicked.connect(self.update_config_file)
        self.convert_to_q_checkbox.clicked.connect(self.set_widget_limits)
        self.wavelength_widget.valueChanged.connect(self.update_config_file)
        self.wavelength_widget.valueChanged.connect(self.set_widget_limits)
        self.plot_xmin_spinbox.valueChanged.connect(self.update_config_file)
        self.plot_xmax_spinbox.valueChanged.connect(self.update_config_file)
        self.plot_data_button.clicked.connect(self.plot_dataset)
        self.algorithm_group.buttonClicked.connect(self.display_algorithm_widgets)
        self.algorithm_group.buttonClicked.connect(self.update_config_file)
        self.calc_err_checkbox.clicked.connect(self.update_config_file)
        self.comp_num_slider.valueChanged.connect(
            lambda value: self.comp_num_label.setText(str(value))
        )
        self.comp_num_slider.valueChanged.connect(self.update_config_file)
        self.comp_num_slider.valueChanged.connect(self.display_reconstruction_widgets)
        self.run_analysis_button.clicked.connect(self.run_analysis)

    def closeEvent(self, event):
        """
        Redefines Qt's closeEvent to shut down entire app if main window closes
        """
        qtw.QApplication.quit()

    def about_button_clicked(self):
        """
        Handles the About Window
        """
        if self.about_window is None:
            self.about_window = AboutDialog(self)
            self.about_window.show()
        else:
            self.about_window.close()
            self.about_window = None

    def display_algorithm_widgets(self):
        """
        Shows/hides widgets based on chosen algorithm
        """
        for widget in (self.PCA_algorithm_widgets +
                       self.NMF_algorithm_widgets +
                       self.ICA_algorithm_widgets +
                       self.SNMF_algorithm_widgets):
            widget.hide()
        match self.algorithm_group.checkedButton().text():
            case "PCA":
                self.PCA_whiten_checkbox.show()
                self.PCA_solver_dropdown.show()
                if self.PCA_solver_dropdown.currentText=='arpack':
                    self.PCA_tol_spinbox.show()
                else:
                    self.PCA_tol_spinbox.hide()
                if self.PCA_solver_dropdown.currentText=='randomized':
                    self.PCA_iterated_power_auto_checkbox.show()
                    self.PCA_iterated_power_spinbox.show()
                    self.PCA_n_oversampled_spinbox.show()
                    self.PCA_power_iteration_normalizer_dropdown.show()
                else:
                    self.PCA_iterated_power_auto_checkbox.hide()
                    self.PCA_iterated_power_spinbox.hide()
                    self.PCA_n_oversampled_spinbox.hide()
                    self.PCA_power_iteration_normalizer_dropdown.hide()
            case "NMF":
                for widget in self.NMF_algorithm_widgets:
                    widget.show()
                if self.NMF_solver_dropdown.currentText=='mu':
                    self.NMF_beta_loss_dropdown.show()
                else:
                    self.NMF_beta_loss_dropdown.hide()
                if self.NMF_solver_dropdown.currentText=='cd':
                    self.NMF_l1_ratio_spinbox.show()
                else:
                    self.NMF_l1_ratio_spinbox.show()
            case "ICA":
                for widget in self.ICA_algorithm_widgets:
                    widget.show()
            case "SNMF":
                for widget in self.SNMF_algorithm_widgets:
                    widget.show()

    def display_reconstruction_widgets(self):
        """
        Displays widgets for reconstructions equaling number of components, and hides the rest
        """
        for widget in self.reconstruct_widgets[:self.comp_num_slider.value()]:
            widget.show()
        for widget in self.reconstruct_widgets[self.comp_num_slider.value():]:
            widget.hide()

    def set_indir(self, indir=None):
        """
        Sets the dataset directory, tests its validity and imports data
        """
        if indir is None or indir is False: # Despite initialized as None, is in reality False from "Load directory" widget
            if self.indir_label.text() == "Select the folder containing your data files":
                indir = str(qtw.QFileDialog.getExistingDirectory(self))
            else:
                indir = str(qtw.QFileDialog.getExistingDirectory(self, directory=self.indir_label.text()))
            if indir == "": # If the operation was cancelled
                return

        # Confirm dataset can be imported by the program
        try:
            self.angles, self.intensities = file_worker.import_dataset(indir)
        except:
            print(f"An error occured when loading ...{'/'.join(indir.split("/")[-3:])}")
        else:
            self.set_widget_limits(new=True)
            self.indir_label.setText(indir)
            self.add_recentdir(indir)
            print("Dataset imported\n")

    def add_recentdir(self, recentdir):
        """
        Adds the relevant directory path to the list of recent dirs
        """
        recentdir_button = qtg.QAction(recentdir, self)
        recentdir_button.triggered.connect(lambda: self.set_indir(recentdir_button.text()))

        if recentdir in (action.text() for action in self.file_submenu_recent_dirs.actions()): # Removes duplicate occurence
            self.file_submenu_recent_dirs.removeAction(next(action for action in self.file_submenu_recent_dirs.actions() if action.text()==recentdir))
        elif len(self.file_submenu_recent_dirs.actions()) >= 10+2: # Deletes oldest recent if there are more than 10
            self.file_submenu_recent_dirs.removeAction(self.file_submenu_recent_dirs.actions()[-3]) # -1 is "clear", -2 is separator, so -3 should be oldest "recent"
        self.file_submenu_recent_dirs.insertAction(self.file_submenu_recent_dirs.actions()[0], recentdir_button)

    def set_bkgfile(self, infile=None):
        """
        Sets the background file, tests its validity and imports it
        """
        if infile is None or infile is False: # Load background widget passes "False"
            if self.bkg_label.text() == "Select a background for subtraction":
                try:
                    infile,_ = qtw.QFileDialog.getOpenFileName(self, directory=self.indir_label.text())
                except:
                    infile,_ = qtw.QFileDialog.getOpenFileName(self)
            else:
                infile,_ = qtw.QFileDialog.getOpenFileName(self, directory=os.path.dirname(self.bkg_label.text()))
            if infile == "": # If the operation was cancelled
                return

        # Confirm bkgfile can be imported by the program
        try:
            file_worker.import_data(infile) # As single-file read is fast, data is not saved as attribute
        except:
            print(f"An error occured when loading {infile.split("/")[-1]}")
        else:
            self.bkg_label.setText(infile)
            self.add_recentbkg(infile)
            print("Background imported\n")

    def add_recentbkg(self, recentbkg):
        """
        Adds the relevant background filepath to the list of recent bkgs
        """
        recentbkg_button = qtg.QAction(recentbkg, self)
        recentbkg_button.triggered.connect(lambda: self.set_bkgfile(recentbkg_button.text()))

        if recentbkg in (action.text() for action in self.file_submenu_recent_bkgs.actions()): # Removes duplicate occurence
            self.file_submenu_recent_bkgs.removeAction(next(action for action in self.file_submenu_recent_bkgs.actions() if action.text()==recentbkg))
        elif len(self.file_submenu_recent_bkgs.actions()) >= 10+2: # Deletes oldest recent if there are more than 10
            self.file_submenu_recent_bkgs.removeAction(self.file_submenu_recent_bkgs.actions()[-3]) # -1 is "clear", -2 is separator, so -3 should be oldest "recent"
        self.file_submenu_recent_bkgs.insertAction(self.file_submenu_recent_bkgs.actions()[0], recentbkg_button)

    def set_widget_limits(self, new=False):
        """
        Sets widget limits based on imported data
        """
        if self.convert_to_q_checkbox.isChecked():
            xmin = funcs.theta_to_q(np.min(self.angles),self.wavelength_widget.value())
            xmax = funcs.theta_to_q(np.max(self.angles),self.wavelength_widget.value())
        else:
            xmin = np.min(self.angles)
            xmax = np.max(self.angles)

        self.plot_xmin_spinbox.setMinimum(xmin)
        self.plot_xmin_spinbox.setMaximum(xmax)
        self.plot_xmax_spinbox.setMinimum(xmin)
        self.plot_xmax_spinbox.setMaximum(xmax)

        if new:
            self.plot_xmin_spinbox.setValue(self.plot_xmin_spinbox.minimum())
            self.plot_xmax_spinbox.setValue(self.plot_xmax_spinbox.maximum())

        for widget in self.reconstruct_widgets:
            widget.setMaximum(len(self.angles))

    def read_config_file(self):
        """
        Reads BaSSET config file to set widget values
        """
        with open(self.configfile, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            for line in lines:
                line = line.replace('\n','')
                pairs = line.split(": ")
                name = pairs[0]
                value = pairs[1]
                if value != "":
                    match name:
                        case "Current file directory":
                            if os.path.exists(value):
                                self.indir_label.setText(value)
                                self.set_indir(value)
                            else:
                                print(f"{value} is not a valid directory")
                        case "Recent directories":
                            for indir in reversed(value.split(", ")):
                                if indir != "":
                                    self.add_recentdir(indir)
                        case "Background file":
                            self.bkg_label.setText(value)
                        case "Background scale":
                            self.bkg_scale_spinbox.setValue(float(value))
                        case "Recent backgrounds":
                            for bkg in reversed(value.split(", ")):
                                if bkg != "":
                                    self.add_recentbkg(bkg)
                        case "Input format":
                            for button in self.input_format_group.buttons():
                                if value == button.text():
                                    button.setChecked(True)
                            if value not in [button.text() for button in self.input_format_group.buttons()]:
                                print(f"Did not recognize \"{value}\" as an input format.\n" \
                                      "Using default input format")
                        case "Convert to Q":
                            if value=="True":
                                self.convert_to_q_checkbox.setChecked(True)
                            elif value=="False":
                                self.convert_to_q_checkbox.setChecked(False)
                            else:
                                print("Did not recognize {line} as True/False")
                        case "Wavelength":
                            self.wavelength_widget.blockSignals(True) # Ensures .setValue doesn't trigger valueChanged
                            self.wavelength_widget.setValue(float(value))
                            self.wavelength_widget.blockSignals(False)
                        case "X-axis range":
                            xmin, xmax = value.split(",")
                            self.plot_xmin_spinbox.setValue(float(xmin))
                            self.plot_xmax_spinbox.setValue(float(xmax))
                        case "Algorithm":
                            for button in self.algorithm_group.buttons():
                                if value == button.text():
                                    button.setChecked(True)
                            if value not in [button.text() for button in self.algorithm_group.buttons()]:
                                print(f"Did not recognize \"{value}\" as an algorithm\n" \
                                      "Using default algorithm")
                        case "Number of components":
                            self.comp_num_slider.blockSignals(True) # Ensures .setValue doesn't trigger valeuChanged
                            self.comp_num_slider.setValue(int(value))
                            self.comp_num_label.setText(value)
                            self.comp_num_slider.blockSignals(False)
                        case "Calculate errors":
                            if value=="True":
                                self.calc_err_checkbox.setChecked(True)
                            elif value=="False":
                                self.calc_err_checkbox.setChecked(False)
                            else:
                                print("Did not recognize {line} as True/False")
                        case _:
                            print(f"\"{line}\" could not be parsed\n" \
                                  "Ignoring.")

    def update_config_file(self):
        """
        Updates BaSSET config file based on widget values
        """
        with open(self.configfile, "w", encoding='utf-8') as outfile:
            outfile.write(f"Current file directory: {self.indir_label.text() if self.indir_label.text()!='Select the folder containing your data files' else ''}\n")
            outfile.write(f"Recent directories: {', '.join(action.text() for action in self.file_submenu_recent_dirs.actions()[:-2])}\n")
            outfile.write(f"Background file: {self.bkg_label.text() if self.bkg_label.text()!='Select a background for subtraction' else ''}\n")
            outfile.write(f"Background scale: {self.bkg_scale_spinbox.value():.3f}\n")
            outfile.write(f"Recent backgrounds: {', '.join(action.text() for action in self.file_submenu_recent_bkgs.actions()[:-2])}\n")
            outfile.write(f"Input format: {self.input_format_group.checkedButton().text()}\n")
            outfile.write(f"Convert to Q: {'True' if self.convert_to_q_checkbox.isChecked() else 'False'}\n")
            outfile.write(f"Wavelength: {self.wavelength_widget.value():.6f}\n")
            outfile.write(f"X-axis range: {self.plot_xmin_spinbox.value():.2f},{self.plot_xmax_spinbox.value():.2f}\n")
            outfile.write(f"Algorithm: {self.algorithm_group.checkedButton().text()}\n")
            outfile.write(f"Number of components: {self.comp_num_slider.value()}\n")
            outfile.write(f"Calculate errors: {'True' if self.calc_err_checkbox.isChecked() else 'False'}\n")

    def plot_dataset(self):
        """
        Plots the input dataset as a waterfall plot
        """
        print("Plotting input dataset\n")
        try:
            if self.angles is None or self.intensities is None:
                self.angles, self.intensities = file_worker.import_dataset(self.indir_label.text())
        except ValueError: # If init as array with more than one elements, truth value is ambiguous
            if self.angles.all() is None or self.intensities.all() is None:
                self.angles, self.intensities = file_worker.import_dataset(self.indir_label.text())

        angles = self.angles.copy()
        intensities = self.intensities.copy()

        if self.convert_to_q_checkbox.isChecked():
            xlabel = "Q [Å⁻¹]"
            angles = funcs.theta_to_q(angles, self.wavelength_widget.value())
        else:
            xlabel = self.input_format_group.checkedButton().text()

        if self.bkg_label.text() != "Select a background for subtraction":
            try:
                _, bkgintensity = file_worker.import_data(self.bkg_label.text())
            except:
                print("! Could not read background." \
                      "Fix/remove background and try again")
                return
            try:
                for i, intensity in enumerate(intensities):
                    intensities[i] = intensity - bkgintensity*self.bkg_scale_spinbox.value()
            except:
                print("! Could not subtract background from data." \
                      "Fix/remove background and try again")
                return

        fig = plt.figure()
        cmap = plt.get_cmap('inferno')
        colors = cmap(np.linspace(0.8, 0, len(intensities)))
        stagger_factor = np.max(intensities) / (15*len(intensities))
        stagger_max = 0

        for i in range(len(intensities)-1, -1, -1):
            yaxis = intensities[i]+i*stagger_factor
            plt.plot(angles[i], yaxis, color=colors[i])
            stagger_max = max(stagger_max,np.max(yaxis))
        plt.xlabel(xlabel)
        plt.ylabel("Intensity (staggered) [A.U.]")
        plt.xlim(max(self.plot_xmin_spinbox.value(), min(angles[i])),
                 min((self.plot_xmax_spinbox.value(), max(angles[i]))))
        if self.input_format_group.checkedButton().text()=="r [Å]":
            plt.ylim(np.min(intensities)*1.005, stagger_max*1.005)
        else:
            plt.ylim(np.min(intensities), stagger_max*1.005)
        plt.ticklabel_format(axis="y", style="sci", scilimits=[0,0])

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
            if not os.path.exists(f"{self.indir_label.text()}/BaSSET_results"):
                os.mkdir(f"{self.indir_label.text()}/BaSSET_results")
            fig.savefig(f"{self.indir_label.text()}/BaSSET_results/input_dataset.png")

    def run_analysis(self):
        """
        Performs necessary pre-processing,
        passes analysis to algorithms,
        and sends results to plotting
        """
        print("Beginning analysis...")
        comp_num = self.comp_num_slider.value()

        try:
            if self.angles is None or self.intensities is None:
                self.angles, self.intensities = file_worker.import_dataset(self.indir_label.text())
        except ValueError: # If init as array with more than one elements, truth value is ambiguous
            if self.angles.all() is None or self.intensities.all() is None:
                self.angles, self.intensities = file_worker.import_dataset(self.indir_label.text())

        angles = self.angles.copy()
        intensities = self.intensities.copy()

        if self.bkg_label.text() != "Select a background for subtraction":
            try:
                _, bkgintensity = file_worker.import_data(self.bkg_label.text())
            except:
                print("! Could not read background." \
                      "Fix/remove background and try again")
                return
            try:
                for i, intensity in enumerate(intensities):
                    intensities[i] = intensity - bkgintensity*self.bkg_scale_spinbox.value()
            except:
                print("! Could not subtract background from data." \
                      "Fix/remove background and try again")
                return

        if self.convert_to_q_checkbox.isChecked():
            angles = funcs.theta_to_q(angles, self.wavelength_widget.value())

        # Crop according to converted data
        if self.plot_xmin_spinbox.value() != self.plot_xmax_spinbox.value():
            xmin_index = np.searchsorted(angles[0], self.plot_xmin_spinbox.value(), side='left')
            xmax_index = np.searchsorted(angles[0], self.plot_xmax_spinbox.value(), side='right')
            angles = angles[:,xmin_index:xmax_index]
            intensities = intensities[:,xmin_index:xmax_index]

        errors = None
        lift_factor = None
        stretch = None
        match self.algorithm_group.checkedButton().text():
            case "PCA":
                fitted, transformed, reconstructed = analysis.PCA_analysis(
                    intensities,
                    comp_num,
                    whiten=self.PCA_whiten_checkbox.isChecked(),
                    svd_solver=self.PCA_solver_dropdown.currentText(),
                    tol=self.PCA_tol_spinbox.value(),
                    iterated_power='auto' if self.PCA_iterated_power_auto_checkbox.isChecked else self.PCA_iterated_power_spinbox.value(),
                    n_oversamples=self.PCA_n_oversampled_spinbox.value(),
                    power_iteration_normalizer=self.PCA_power_iteration_normalizer_dropdown.currentText()
                )
            case "NMF":
                fitted, transformed, reconstructed, errors, lift_factor = analysis.NMF_analysis(
                    intensities,
                    comp_num,
                    init=self.NMF_init_dropdown.currentText(),
                    solver=self.NMF_solver_dropdown.currentText(),
                    beta_loss=self.NMF_beta_loss_dropdown.currentText() if self.NMF_solver_dropdown.currentText()=="mu" else "frobenius",
                    tol=self.NMF_tol_spinbox.value(),
                    max_iter=self.NMF_max_iter_spinbox.value(),
                    alpha_W=self.NMF_alpha_W_spinbox.value(),
                    alpha_H='same' if self.NMF_alpha_H_same_checkbox.isChecked() else self.NMF_alpha_H_spinbox.value(),
                    l1_ratio=self.NMF_l1_ratio_spinbox.value(),
                    calc_err=self.calc_err_checkbox.isChecked(),
                    rescale=self.NMF_rescale_checkbox.isChecked()
                )
            case "ICA":
                fitted, transformed, reconstructed = analysis.ICA_analysis(
                    intensities,
                    comp_num,
                    algorithm=self.ICA_algorithm_dropdown.currentText(),
                    whiten=False if self.ICA_whiten_dropdown.currentText()=='False' else self.ICA_whiten_dropdown.currentText(),
                    fun=self.ICA_fun_dropdown.currentText(),
                    max_iter=self.ICA_max_iter_spinbox.value(),
                    tol=self.ICA_tol_spinbox.value(),
                    whiten_solver=self.ICA_whiten_solver_dropdown.currentText(),
                    calc_err=self.calc_err_checkbox.isChecked()
                )
            case "SNMF":
                fitted, transformed, reconstructed, errors, stretch = analysis.SNMF_analysis(
                    intensities,
                    comp_num,
                    min_iter=self.SNMF_min_iter_spinbox.value(),
                    max_iter=self.SNMF_max_iter_spinbox.value(),
                    tol=self.SNMF_tol_spinbox.value(),
                    rho=self.SNMF_rho_spinbox.value(),
                    eta=self.SNMF_eta_spinbox.value(),
                    calc_err=self.calc_err_checkbox.isChecked()
                )

        print("Analysis completed\n")
        self.plot_analysis(comp_num, angles, intensities, fitted, transformed, reconstructed, errors, lift_factor, stretch)

    def plot_analysis(self, comp_num, angles, intensities, fitted, transformed, reconstructed, errors=None, lift_factor=None, stretch=None):
        """
        Plots analysis results
        """
        if self.convert_to_q_checkbox.isChecked():
            xlabel = "Q [Å⁻¹]"
        else:
            xlabel = self.input_format_group.checkedButton().text()

        recon_num = []
        for widget in self.reconstruct_widgets:
            if widget.isVisible():
                recon_num.append(widget.value()-1)
        if any(x>len(reconstructed) for x in recon_num):
            print("Reconstruction scan larger than number of scans." \
                  "Defaulting to uniform distribution")
            recon_num[0]=-1

        if any(x==-1 for x in recon_num): # any reconstuction widgets are 0, is -1 due to line above
            recon_num = list(np.linspace(0, len(reconstructed), comp_num-1, endpoint=False, dtype=int))
            recon_num.append(len(reconstructed)-1) # append end-point to get first and last scan

        cmap = plt.get_cmap('inferno')
        colors = cmap(np.linspace(0, 0.8, comp_num))

        fig = plt.figure()
        gs = fig.add_gridspec(3, comp_num)
        if (comp_num % 2) == 0:
            ax_scores = fig.add_subplot(gs[2,0:(comp_num // 2)])
            ax_errors = fig.add_subplot(gs[2,(comp_num // 2):])
        else:
            ax_scores = fig.add_subplot(gs[2,0:(comp_num // 2)+1])
            ax_errors = fig.add_subplot(gs[2,(comp_num // 2)+1:])

        for i in range(comp_num):
            ax_comp = fig.add_subplot(gs[0, i])
            ax_recon = fig.add_subplot(gs[1, i])
            ax_comp.plot(angles[i], fitted.components_[i], "k")
            ax_comp.set_title(f"Component {i+1}")
            ax_comp.set_xlabel(xlabel)
            ax_comp.set_xlim(np.min(angles[i]), np.max(angles[i]))
            ax_comp.set_ylim(min(fitted.components_[i]) - max(fitted.components_[i])*0.05,
                             max(fitted.components_[i])*1.05
            )

            difference = intensities[recon_num[i]]-reconstructed[recon_num[i]]
            distance = np.abs(min(np.min(reconstructed[recon_num[i]]),
                                  np.min(intensities[recon_num[i]]))-np.max(difference)
            )

            ax_recon.plot(angles[recon_num[i]], reconstructed[recon_num[i]], color="#DD0000", label="Reconstructed")
            ax_recon.plot(angles[recon_num[i]], intensities[recon_num[i]], "k:", label="Original")
            ax_recon.plot(angles[recon_num[i]], difference-distance*1.05, color="#2EC483", label="Difference")
            ax_recon.ticklabel_format(axis="y", style="sci", scilimits=[0,0])
            ax_recon.set_title(f"Scan {recon_num[i]+1}") # Reconstruction num
            ax_recon.set_xlabel(xlabel)
            ax_recon.sharex(ax_comp)
            ax_recon.set_ylim((np.min(difference)-distance)*1.05 - max(np.max(intensities[recon_num[i]]),np.max(reconstructed[recon_num[i]]))*0.05,
                              max(np.max(intensities[recon_num[i]]),np.max(reconstructed[recon_num[i]]))*1.05)

            ax_scores.plot(np.arange(1,len(angles)+1), transformed[:,i], color=colors[i], label=f"Comp. {i+1}")

            if i==0:
                ax_comp_master = ax_comp
            else:
                ax_comp.sharex(ax_comp_master)

        for i, recon in enumerate(recon_num):
            ax_scores.axvline(recon+1, color="k", linestyle=":", label="Recons" if i==0 else '')
        ax_scores.ticklabel_format(axis="y", style="sci", scilimits=[0,0])
        ax_scores.legend()
        l, h = ax_recon.get_legend_handles_labels()
        ax_errors.legend(l, h, title="↑", frameon=True)
        ax_scores.set_xlabel("Scan #")
        ax_scores.set_xlim(1, len(angles))

        if self.NMF_rescale_checkbox.isChecked() and self.algorithm_group.checkedButton().text()=="NMF":
            ax_scores.set_title("Normalized Scores")
            ax_scores.set_ylim(-0.01, 1.01)
        else:
            ax_scores.set_title("Scores")
            ax_scores.set_ylim(min(0,np.min(transformed)), np.max(transformed))

        match self.algorithm_group.checkedButton().text():
            case "PCA":
                ax_errors.plot(np.arange(1, min(10,np.shape(intensities))+1), fitted.explained_variance_ratio_*100, "ko--")
                ax_errors.set_title("Explained variances")
            case "NMF":
                if errors is not None:
                    ax_errors.plot(np.arange(1, min(10,np.shape(intensities))+1), errors, "ko--")
                    ax_errors.set_title("Reconstruction error")
            case "ICA":
                pass
            case "SNMF":
                pass
                """
                ax_errors.plot(np.arange(1, min(10,min(np.shape(intensities)))+1), errors, "ko--")
                ax_errors.set_title("Reconstruction error (Not Implemented)")
                axs[2][2].plot(np.arange(1, min(10,min(np.shape(intensities)))+1, stretch, "ko--"))
                axs[2][2].set_title("Stretching")
                """
        ax_errors.set_xlim(0.9, 10.1)

        ax_errors.set_xlabel("# of Components")

        fig.canvas.manager.set_window_title(f"{self.algorithm_group.checkedButton().text()} ({comp_num}): x:({self.plot_xmin_spinbox.value()},{self.plot_xmax_spinbox.value()})")

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
            self.export_results(angles, intensities, fitted, transformed, reconstructed, fig, errors, lift_factor, stretch)

    def write_summary(self, results_path, errors=None, lift_factor=None):
        """
        Writes summary of analysis
        """
        with open(f"{results_path}/summary.txt", "w", encoding='utf-8') as outfile:
            outfile.write(f"Summary of {self.comp_num_slider.value()} component {self.algorithm_group.checkedButton().text()} analysis of data in \"...{self.indir_label.text()[-51:]}\"\n\n")
            outfile.write(f"Data was analyzed from {self.plot_xmin_spinbox.value()} to {self.plot_xmax_spinbox.value()} {self.input_format_group.checkedButton().text()}\n")
            outfile.write("The following algorithm parameters where used:\n")
            match self.algorithm_group.checkedButton().text():
                case "PCA":
                    outfile.write(f"Whiten: {self.PCA_whiten_checkbox.isChecked()}\n")
                    outfile.write(f"SVD solver: {self.PCA_solver_dropdown.currentText()}\n")
                    if self.PCA_solver_dropdown.currentText()=="arpack":
                        outfile.write(f"Tolerance: {self.PCA_tol_spinbox.value()}\n")
                    if self.PCA_solver_dropdown.currentText()=="randomized":
                        outfile.write(f"# of iterations (Power method): {self.PCA_iterated_power_spinbox() if self.PCA_iterated_power_auto_checkbox.isChecked() else 'auto'}\n")
                        outfile.write(f"Additional vectors to sample data: {self.PCA_n_oversampled_spinbox.value()}\n")
                        outfile.write(f"Power iteration normalizer: {self.PCA_power_iteration_normalizer_dropdown.currentText()}\n")
                        outfile.write(f"\nThe PCA explained variance for {self.comp_num_slider.value()} components was: {errors[self.comp_num_slider.value()+1]}\n")
                case "NMF":
                    outfile.write(f"Initialization method: {self.NMF_init_dropdown.currentText()}\n")
                    outfile.write(f"Numerical solver: {self.NMF_solver_dropdown.currentText()}\n")
                    if self.NMF_solver_dropdown.currentText()=='mu':
                        outfile.write(f"Beta divergence to minimize: {self.NMF_beta_loss_dropdown.currentText()}\n")
                    outfile.write(f"Tolerance: {self.NMF_tol_spinbox.value()}\n")
                    outfile.write(f"Maximum number of iterations: {self.NMF_max_iter_spinbox.value()}\n")
                    outfile.write(f"Regularization constant for features: {self.NMF_alpha_W_spinbox.value()}\n")
                    outfile.write(f"Regularization constant for samples: {self.NMF_alpha_H_spinbox.value()}\n") # Should work even if alpha_Hsame=True
                    outfile.write(f"Regularization mixing parameter (0=l2, 1=l1): {self.NMF_l1_ratio_spinbox.value()}\n")
                    if lift_factor<0:
                        outfile.write(f"Due to negative values, your dataset was lifted by {lift_factor} during analysis, before subsequent lowering\n")
                    if self.NMF_rescale_checkbox:
                        outfile.write("Scores were rescaled to sum to 1\n")
                    if errors is not None:
                        outfile.write(f"\nThe NMF reconstruction error for {self.comp_num_slider.value()} components was: {errors[self.comp_num_slider.value()+1]:.2f}\n")
                case "ICA":
                    outfile.write(f"Algorithm: {self.ICA_algorithm_dropdown.currentText()}\n")
                    outfile.write(f"Whitening strategy: {self.ICA_whiten_dropdown.currentText()}\n")
                    outfile.write(f"Whitening solver: {self.ICA_whiten_solver_dropdown.currentText()}\n")
                    outfile.write(f"Functional G form function: {self.ICA_fun_dropdown.currentText()}\n")
                    outfile.write(f"Maximum number of iterations: {self.ICA_max_iter_spinbox.value()}\n")
                    outfile.write(f"Tolerance: {self.ICA_tol_spinbox.value()}\n")
                case "SNMF":
                    outfile.write(f"Minimum number of iterations: {self.SNMF_min_iter_spinbox.value()}\n")
                    outfile.write(f"Maximum number of iterations: {self.SNMF_max_iter_spinbox.value()}\n")
                    outfile.write(f"Tolerance: {self.SNMF_tol_spinbox.value()}\n")
                    outfile.write(f"Stretching factor: {self.SNMF_rho_spinbox.value()}\n")
                    outfile.write(f"Sparsity factor: {self.SNMF_eta_spinbox.value()}\n")
                    outfile.write("! Be aware that stretching uses division, not multiplication (component / stretch) !")
        print("Summary written")

    def export_results(self, angles, intensities, fitted, transformed, reconstructed, fig=None, errors=None, lift_factor=None, stretch=None):
        """
        Handles exporting of results
        """
        print("Exporting results")
        if not os.path.exists(f"{self.indir_label.text()}/BaSSET_results"):
            os.mkdir(f"{self.indir_label.text()}/BaSSET_results")
            print(f"Results can be found in: {self.indir_label.text()}/BaSSET_results")

        export_time = datetime.now().strftime("%y%m%d-%H%M%S")
        results_path = f"{self.indir_label.text()}/BaSSET_results/{export_time}_{self.algorithm_group.checkedButton().text()}_{self.comp_num_slider.value()}"
        os.mkdir(results_path)
        if fig is not None:
            fig.savefig(f"{results_path}/overview.jpg")
        self.write_summary(results_path, errors, lift_factor)
        file_worker.write_components(angles, results_path, fitted)
        file_worker.write_scores(results_path, transformed)
        if self.export_recons_checkbox.isChecked():
            file_worker.write_reconstructions(angles, results_path, reconstructed)
        if self.export_diffs_checkbox.isChecked():
            file_worker.write_differences(angles, intensities, results_path, reconstructed)
        if self.export_comp_contribute_checkbox.isChecked():
            file_worker.write_comp_contribute(angles, results_path, fitted, transformed, stretch)
        if self.algorithm_group.checkedButton().text()=="SNMF":
            file_worker.write_stretch(results_path, stretch)
        print("Exporting results completed\n")
        return None

class AboutDialog(qtw.QDialog):
    """
    Class container for About window
    """

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

        self.version = qtw.QLabel("Version: 1.3.3a")
        self.version.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.version)

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
    """
    Main GUI handler
    """
    app = qtw.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__=="__main__":
    main()
