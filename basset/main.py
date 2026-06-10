"""
BaSSET GUI module
"""

import sys
import os
from datetime import datetime

from basset.utils import (
    analysis,
    file_worker,
    funcs,
    gui_helper
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


class MainWindow(qtw.QMainWindow):
    """
    Main BaSSET GUI window 
    """
    def __init__(self):
        super().__init__()

        self.angles = None
        self.intensities = None
        self.dataset_details = {
            "xmin": None,
            "xmax": None,
            "scanmax": None
        }

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
        self.input_format_title.setSizePolicy(
            qtw.QSizePolicy.Policy.Minimum,
            qtw.QSizePolicy.Policy.Fixed
        )
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
        self.indir_label.setSizePolicy(
            qtw.QSizePolicy.Policy.Preferred,
            qtw.QSizePolicy.Policy.Fixed
        )
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

        self.bkg_scale_spinbox = gui_helper.SciSpinBox()
        self.bkg_scale_spinbox.setMinimum(0)
        self.bkg_scale_spinbox.setValue(1)
        self.bkg_scale_spinbox.setToolTip('Scale your background to match dataset\n' \
                                          '(View in "Plot input dataset")')
        self.bkg_layout.addWidget(self.bkg_scale_spinbox)

        self.getbkg_button = qtw.QPushButton("Load background")
        self.getbkg_button.setSizePolicy(qtw.QSizePolicy.Policy.Fixed, qtw.QSizePolicy.Policy.Fixed)
        self.bkg_layout.addWidget(self.getbkg_button)

        self.limit_dataset_layout = qtw.QGridLayout()
        self.grid.addLayout(self.limit_dataset_layout, 2,0)

        self.limit_dataset_label = qtw.QLabel("Limit dataset")
        self.limit_dataset_label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.limit_dataset_label.setSizePolicy(
            qtw.QSizePolicy.Policy.Minimum,
            qtw.QSizePolicy.Policy.Fixed
        )
        self.limit_dataset_layout.addWidget(self.limit_dataset_label, 0,0,1,3)

        self.limit_xaxis_checkbox = qtw.QCheckBox("X-axis")
        self.limit_dataset_layout.addWidget(self.limit_xaxis_checkbox, 1,0)
        self.limit_xaxis_checkbox.setToolTip("Set limits to x-axis for analysis")

        self.xmin_spinbox = qtw.QDoubleSpinBox()
        self.xmin_spinbox.setSingleStep(0.01)
        self.limit_dataset_layout.addWidget(self.xmin_spinbox, 1,1)
        self.xmin_spinbox.setToolTip("Minimum x-value to include in analysis")

        self.xmax_spinbox = qtw.QDoubleSpinBox()
        self.xmax_spinbox.setSingleStep(0.01)
        self.limit_dataset_layout.addWidget(self.xmax_spinbox, 1,2)
        self.xmax_spinbox.setToolTip("Maximum x-value to include in analysis")

        self.limit_xaxis_checkbox.toggled.connect(
            lambda value: (
                self.xmin_spinbox.setEnabled(value),
                self.xmax_spinbox.setEnabled(value)
            )
        )

        self.limit_scans_checkbox = qtw.QCheckBox("Scans")
        self.limit_dataset_layout.addWidget(self.limit_scans_checkbox, 2,0)
        self.limit_scans_checkbox.setToolTip("Set limits to scans for analysis")

        self.scanmin_spinbox = qtw.QSpinBox()
        self.scanmin_spinbox.setMinimum(1)
        self.limit_dataset_layout.addWidget(self.scanmin_spinbox, 2,1)
        self.scanmin_spinbox.setToolTip("Minimum scan number to include in analysis")

        self.scanmax_spinbox = qtw.QSpinBox()
        self.scanmax_spinbox.setMinimum(1)
        self.limit_dataset_layout.addWidget(self.scanmax_spinbox, 2,2)
        self.scanmax_spinbox.setToolTip("Maximum scan number to include in analysis")

        self.limit_scans_checkbox.toggled.connect(
            lambda value: (
                self.scanmin_spinbox.setEnabled(value),
                self.scanmax_spinbox.setEnabled(value)
            )
        )

        self.plot_data_button = qtw.QPushButton("Plot input dataset")
        self.plot_data_button.setSizePolicy(
            qtw.QSizePolicy.Policy.MinimumExpanding,
            qtw.QSizePolicy.Policy.Ignored
        )
        self.grid.addWidget(self.plot_data_button, 3,0)

        self.exp_win_layout = qtw.QGridLayout()
        self.grid.addLayout(self.exp_win_layout, 4,0)

        self.exp_win_label = qtw.QLabel("Expanding window fitting")
        self.exp_win_label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.exp_win_label.setSizePolicy(
            qtw.QSizePolicy.Policy.Minimum,
            qtw.QSizePolicy.Policy.Fixed
        )
        self.exp_win_layout.addWidget(self.exp_win_label, 0,0,1,3)

        self.exp_win_enable_checkbox = qtw.QCheckBox("Enable")
        self.exp_win_enable_checkbox.setToolTip(
            "Set own windows or use uniform distribution\n"
            "(incompatible with \"Calculate errors\")"
        )
        self.exp_win_layout.addWidget(self.exp_win_enable_checkbox,1,0)


        self.exp_win_custom_checkbox = qtw.QCheckBox("Custom")
        self.exp_win_custom_checkbox.setDisabled(True)
        self.exp_win_custom_checkbox.setToolTip("Set own windows or use uniform distribution")
        self.exp_win_layout.addWidget(self.exp_win_custom_checkbox,1,1)

        # IF UNIFROM DISTRIBUTION
        self.exp_win_num_spinbox = qtw.QSpinBox()
        self.exp_win_num_spinbox.setMinimum(1)
        self.exp_win_num_spinbox.setDisabled(True)
        self.exp_win_num_spinbox.setToolTip("Number of windows for uniform distribution")
        self.exp_win_layout.addWidget(self.exp_win_num_spinbox,1,2)

        self.exp_win_comps_label = qtw.QLabel("Components")
        self.exp_win_layout.addWidget(self.exp_win_comps_label,2,0)

        self.exp_win_comps_line = qtw.QLineEdit()
        self.exp_win_comps_line.setDisabled(True)
        self.exp_win_comps_line.setToolTip(
            "Enter how many components you'd like for each window\n"
            "(leave empty for all components from the start)"
        )
        self.exp_win_layout.addWidget(self.exp_win_comps_line, 2,1,1,2)

        self.exp_win_custom_label = qtw.QLabel("Windows")
        self.exp_win_layout.addWidget(self.exp_win_custom_label,3,0)

        self.exp_win_custom_line = qtw.QLineEdit()
        self.exp_win_custom_line.setDisabled(True)
        self.exp_win_custom_line.setToolTip("Enter window borders separated by comma (,)")
        self.exp_win_layout.addWidget(self.exp_win_custom_line,3,1,1,2)

        self.exp_win_enable_checkbox.toggled.connect(
            lambda value: (
                self.exp_win_custom_checkbox.setEnabled(value),
                self.exp_win_comps_line.setEnabled(True),
                self.exp_win_custom_checkbox.toggled.emit(self.exp_win_custom_checkbox.isChecked())
            )
        )
        self.exp_win_custom_checkbox.toggled.connect(
            lambda value: (
                self.exp_win_custom_line.setEnabled(value),
                self.exp_win_num_spinbox.setEnabled(not value)
            )
        )

        #############################
        ##### Algorithm widgets #####
        #############################
        self.algorithm_layout = qtw.QGridLayout()
        self.grid.addLayout(self.algorithm_layout, 2,1)

        self.algorithm_group = qtw.QButtonGroup()
        self.algorithm_title = qtw.QLabel("Algorithm")
        self.algorithm_title.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.algorithm_title.setSizePolicy(
            qtw.QSizePolicy.Policy.Minimum,
            qtw.QSizePolicy.Policy.Fixed
        )
        self.algorithm_layout.addWidget(self.algorithm_title, 0,0,1,3)

        self.pca_button = qtw.QRadioButton("PCA")
        self.pca_button.setChecked(True)
        self.pca_button.setToolTip("Principal Component Analysis (scikit-learn)")
        self.algorithm_layout.addWidget(self.pca_button, 1,0)

        self.nmf_button = qtw.QRadioButton("NMF")
        self.nmf_button.setToolTip("Non-Negative Matrix Factorization (scikit-learn)")
        self.algorithm_layout.addWidget(self.nmf_button, 1,1)

        self.ica_button = qtw.QRadioButton("ICA")
        self.ica_button.setToolTip("Independent Component Analysis (scikit-learn)")
        self.algorithm_layout.addWidget(self.ica_button, 1,2)

        self.snmf_button = qtw.QRadioButton("SNMF")
        self.snmf_button.setToolTip("Stretched Non-Negative Matrix Factorization (diffpy)\n"
                                   "WARNING: The SNMF algorithm takes a lot of time.\n"
                                   "Error calculation will not be performed.")
        self.algorithm_layout.addWidget(self.snmf_button, 2,0)

        self.calc_err_checkbox = qtw.QCheckBox("Calculate errors")
        self.calc_err_checkbox.setChecked(True)
        self.calc_err_checkbox.setToolTip("Calculates errors for 1 to 10 components")
        self.algorithm_layout.addWidget(self.calc_err_checkbox, 2,1,1,2)

        self.algorithm_group.addButton(self.pca_button)
        self.algorithm_group.addButton(self.nmf_button)
        self.algorithm_group.addButton(self.ica_button)
        self.algorithm_group.addButton(self.snmf_button)

        self.comp_num_layout = qtw.QGridLayout()
        self.grid.addLayout(self.comp_num_layout, 3,1)

        self.comp_num_title = qtw.QLabel("Number of Components")
        self.comp_num_title.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.comp_num_title.setSizePolicy(
            qtw.QSizePolicy.Policy.Minimum,
            qtw.QSizePolicy.Policy.Fixed
        )
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
        self.comp_num_slider.setSizePolicy(qtw.QSizePolicy.Policy.Minimum,
                                           qtw.QSizePolicy.Policy.Fixed
        )
        self.num_components_slider_layout.addWidget(self.comp_num_slider)

        self.comp_num_label = qtw.QLabel(str(self.comp_num_slider.value()))
        self.comp_num_label.setSizePolicy(
            qtw.QSizePolicy.Policy.Fixed,
            qtw.QSizePolicy.Policy.Fixed
        )
        self.num_components_slider_layout.addWidget(self.comp_num_label)


        self.algorithm_parameters_layout = qtw.QGridLayout()
        self.grid.addLayout(self.algorithm_parameters_layout, 4,1)

        self.algorithm_parameters_title = qtw.QLabel("Algorithm Parameters")
        self.algorithm_parameters_title.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.algorithm_parameters_title.setSizePolicy(qtw.QSizePolicy.Policy.Minimum,
                                                      qtw.QSizePolicy.Policy.Fixed
        )
        self.algorithm_parameters_layout.addWidget(self.algorithm_parameters_title, 0,0, 1,3)

        ##########################
        ##### PCA parameters #####
        ##########################
        self.pca_whiten_checkbox = qtw.QCheckBox("Whiten")
        self.algorithm_parameters_layout.addWidget(self.pca_whiten_checkbox, 1,0)
        self.pca_whiten_checkbox.setToolTip("[whiten]\n" \
        "Whitening will remove some information from the transformed signal\n" \
        "(the relative variance scales of the components) but can sometime\n" \
        "improve the predictive accuracy of the downstream estimators\n" \
        "by making their data respect some hard-wired assumptions")

        self.pca_solver_dropdown = qtw.QComboBox()
        self.pca_solver_dropdown.addItems([
            'auto',
            'full',
            'covariance_eigh',
            'arpack',
            'randomized'
        ])
        self.algorithm_parameters_layout.addWidget(self.pca_solver_dropdown, 1,1)
        self.pca_solver_dropdown.setToolTip("[svd_solver]\n" \
        "The type of Singular Value Decomposition solver to use:\n" \
        "auto (default): Chooses solver based on size of dataset and number of components\n" \
        "full: Runs exact full SVD\n" \
        "covariance_eigh: Precomputes covarience for eigenvalue decompositon.\n" \
        "    Efficient for many scans of few datapoints (rare for scattering)\n" \
        "arpack: Runs SVD truncated to number of components.\n" \
        "    Requires fewer components than number of scans\n" \
        "randomized: Runs randomized SVD")

        # Only for 'arpack' solver
        self.pca_tol_spinbox = gui_helper.SciSpinBox()
        self.pca_tol_spinbox.setMinimum(0)
        self.pca_tol_spinbox.setValue(0)
        self.algorithm_parameters_layout.addWidget(self.pca_tol_spinbox, 1,2)
        self.pca_tol_spinbox.setToolTip("[tol]\n" \
        "Tolerance for singular values using 'arpack'")
        self.pca_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.pca_tol_spinbox.show()
            if currentText=='arpack'
            else self.pca_tol_spinbox.hide()
        )

        # Only for 'randomized' solver
        self.pca_iterated_power_spinbox = qtw.QSpinBox()
        self.pca_iterated_power_spinbox.setMinimum(0)
        self.pca_iterated_power_spinbox.setMaximum(999999)
        self.pca_iterated_power_spinbox.setValue(0)
        self.pca_iterated_power_spinbox.setSingleStep(100)
        self.algorithm_parameters_layout.addWidget(self.pca_iterated_power_spinbox, 2,0)
        self.pca_iterated_power_spinbox.setToolTip("[iterated_power]\n" \
        "Number of iterations for the power method in 'randomized'")
        self.pca_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.pca_iterated_power_spinbox.show()
            if currentText=='randomized'
            else self.pca_iterated_power_spinbox.hide()
        )

        # Only for 'randomized' solver
        self.pca_iterated_power_auto_checkbox = qtw.QCheckBox("auto")
        self.algorithm_parameters_layout.addWidget(self.pca_iterated_power_auto_checkbox, 2,1)
        self.pca_iterated_power_auto_checkbox.setToolTip("[iterated_power]\n" \
        "Automatically choose number of iterations")
        self.pca_iterated_power_auto_checkbox.clicked.connect(
            lambda state: self.pca_iterated_power_spinbox.setDisabled(True)
            if state
            else self.pca_iterated_power_spinbox.setDisabled(False)
        )
        self.pca_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.pca_iterated_power_auto_checkbox.show()
            if currentText=='randomized'
            else self.pca_iterated_power_auto_checkbox.hide()
        )

        # Only for 'randomized' solver
        self.pca_n_oversampled_spinbox = qtw.QSpinBox()
        self.pca_n_oversampled_spinbox.setMinimum(0)
        self.pca_n_oversampled_spinbox.setMaximum(50)
        self.pca_n_oversampled_spinbox.setValue(10)
        self.pca_n_oversampled_spinbox.setSingleStep(1)
        self.algorithm_parameters_layout.addWidget(self.pca_n_oversampled_spinbox, 2,2)
        self.pca_n_oversampled_spinbox.setToolTip("[n_oversamples]\n" \
        "Additional number of random vectors to sample using 'randomized'")
        self.pca_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.pca_n_oversampled_spinbox.show()
            if currentText=='randomized'
            else self.pca_n_oversampled_spinbox.hide()
        )

        # Only for 'randomized' solver // Not for 'arpack' solver
        self.pca_power_iteration_normalizer_dropdown = qtw.QComboBox()
        self.pca_power_iteration_normalizer_dropdown.addItems(['auto', 'QR', 'LU', 'none'])
        self.algorithm_parameters_layout.addWidget(self.pca_power_iteration_normalizer_dropdown,1,2)
        self.pca_power_iteration_normalizer_dropdown.setToolTip("[power_iteration_normalizer]\n" \
        "Power iteration normalizer using 'randomized'")
        self.pca_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.pca_power_iteration_normalizer_dropdown.show()
            if currentText=='randomized'
            else self.pca_power_iteration_normalizer_dropdown.hide()
        )

        # Ignored PCA parameters:
        # random_state

        self.pca_algorithm_widgets = [
            self.pca_whiten_checkbox, self.pca_solver_dropdown,
            self.pca_tol_spinbox, self.pca_iterated_power_spinbox,
            self.pca_iterated_power_auto_checkbox, self.pca_n_oversampled_spinbox,
            self.pca_n_oversampled_spinbox, self.pca_power_iteration_normalizer_dropdown
        ]

        ##########################
        ##### NMF parameters #####
        ##########################
        self.nmf_init_dropdown = qtw.QComboBox()
        self.nmf_init_dropdown.addItems(["nndsvda", "random", "nndsvd", "nndsvdar"])
        self.algorithm_parameters_layout.addWidget(self.nmf_init_dropdown, 1,0)
        self.nmf_init_dropdown.setToolTip("[init]\n" \
        "Method used to initialize the procedure:\n" \
        "('nndsvda' is recommended for PDF and 'nndsvd' is recommended for XRD)\n" \
        "nndsvda (default): Better when sparsity is not desired (PDF)\n" \
        "random: Random non-negative matrices\n" \
        "nndsvd: Non-negative Double Singular Value Decomposition (better for sparseness) (XRD)\n" \
        "nndsvdar: Faster, less accurate alternative to NNDSVDa when sparsity is not desired")

        self.nmf_solver_dropdown = qtw.QComboBox()
        self.nmf_solver_dropdown.addItems(["cd", "mu"])
        self.algorithm_parameters_layout.addWidget(self.nmf_solver_dropdown, 1,1)
        self.nmf_solver_dropdown.setToolTip("[solver]\n" \
        "Numerical solver to use:\n" \
        "cd (default): Coordinate Descent\n" \
        "mu: Multiplicative Update\n" \
        "('mu' gives poor results with 'nndsvd' as it cannot update zeros in initialization)")

        self.nmf_max_iter_spinbox = qtw.QSpinBox()
        self.nmf_max_iter_spinbox.setMinimum(0)
        self.nmf_max_iter_spinbox.setMaximum(999999)
        self.nmf_max_iter_spinbox.setValue(2500)
        self.algorithm_parameters_layout.addWidget(self.nmf_max_iter_spinbox, 2,0)
        self.nmf_max_iter_spinbox.setToolTip("[max_iter]\n" \
        "Maximum number of iterations before timing out")

        self.nmf_tol_spinbox = gui_helper.SciSpinBox()
        self.nmf_tol_spinbox.setMinimum(0)
        self.nmf_tol_spinbox.setValue(1e-4)
        self.algorithm_parameters_layout.addWidget(self.nmf_tol_spinbox, 2,1)
        self.nmf_tol_spinbox.setToolTip("[tol]\n" \
        "Tolerance of the stopping condition")

        # Used in 'cd' solver
        self.nmf_l1_ratio_spinbox = qtw.QDoubleSpinBox()
        self.nmf_l1_ratio_spinbox.setMinimum(0)
        self.nmf_l1_ratio_spinbox.setMaximum(1)
        self.nmf_l1_ratio_spinbox.setValue(0)
        self.nmf_l1_ratio_spinbox.setSingleStep(0.05)
        self.algorithm_parameters_layout.addWidget(self.nmf_l1_ratio_spinbox, 1,2)
        self.nmf_l1_ratio_spinbox.setToolTip("[l1_ratio]\n" \
        "Regularization mixing parameter:\n" \
        "(0, default): elementwise L2 penalty aka. Frobenius Norm\n"
        "(1): elementwwise L1 penalty (better for sparseness)\n")
        self.nmf_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.nmf_l1_ratio_spinbox.show()
            if currentText=='cd'
            else self.nmf_l1_ratio_spinbox.hide()
        )

        # Only for 'mu' solver
        self.nmf_beta_loss_dropdown = qtw.QComboBox()
        # "itakura-saito" not included. Cannot have zeros, which XRD/PDF often has
        self.nmf_beta_loss_dropdown.addItems(["frobenius", "kullback-leibler"])
        self.algorithm_parameters_layout.addWidget(self.nmf_beta_loss_dropdown, 1,2)
        self.nmf_beta_loss_dropdown.setToolTip("[beta_loss]\n" \
        "Beta divergence to be minimized," \
        "measuring the distance between X and the dot product WH using 'mu':\n" \
        "frobenius is default")
        self.nmf_solver_dropdown.currentTextChanged.connect(
            lambda currentText: self.nmf_beta_loss_dropdown.show()
            if currentText=='mu'
            else self.nmf_beta_loss_dropdown.hide()
        )

        self.nmf_alpha_w_spinbox = qtw.QDoubleSpinBox()
        self.nmf_alpha_w_spinbox.setMinimum(0)
        self.nmf_alpha_w_spinbox.setMaximum(10)
        self.nmf_alpha_w_spinbox.setValue(0)
        self.nmf_alpha_w_spinbox.setDecimals(5)
        self.nmf_alpha_w_spinbox.setSingleStep(0.00005)
        self.algorithm_parameters_layout.addWidget(self.nmf_alpha_w_spinbox, 3,0)
        self.nmf_alpha_w_spinbox.setToolTip(
            "[alpha_W]\n"
            "Constant that multiplies the regularization terms of the mixing of features:\n"
            "(Regularization is a penalty term to constrain parameter"
            "complexity and reduce overfitting)\n"
            "0 (default) means no regularization"
        )

        self.nmf_alpha_h_spinbox = qtw.QDoubleSpinBox()
        self.nmf_alpha_h_spinbox.setMinimum(0)
        self.nmf_alpha_h_spinbox.setMaximum(10)
        self.nmf_alpha_h_spinbox.setValue(0)
        self.nmf_alpha_h_spinbox.setDecimals(5)
        self.nmf_alpha_h_spinbox.setSingleStep(0.00005)
        self.nmf_alpha_h_spinbox.setDisabled(True)
        self.algorithm_parameters_layout.addWidget(self.nmf_alpha_h_spinbox, 3,1)
        self.nmf_alpha_h_spinbox.setToolTip("[alpha_H]\n" \
        "Constant that multiplies the regularization terms of the features:\n" \
        "0 means no regularization. By default same as alpha_W ")

        self.nmf_alpha_h_same_checkbox = qtw.QCheckBox("same")
        self.nmf_alpha_h_same_checkbox.setChecked(True)
        self.algorithm_parameters_layout.addWidget(self.nmf_alpha_h_same_checkbox, 3,2)
        self.nmf_alpha_h_same_checkbox.setToolTip("[alpha_H]\n" \
        "Use same regularization constant for features and mixing")
        self.nmf_alpha_h_same_checkbox.clicked.connect(
            lambda state: self.nmf_alpha_h_spinbox.setDisabled(True)
            if state
            else self.nmf_alpha_h_spinbox.setDisabled(False)
        )
        self.nmf_alpha_h_same_checkbox.clicked.connect(
            lambda state: self.nmf_alpha_h_spinbox.setValue(self.nmf_alpha_w_spinbox.value())
            if state
            else None
        )
        self.nmf_alpha_w_spinbox.valueChanged.connect(
            lambda value: self.nmf_alpha_h_spinbox.setValue(value)
            if self.nmf_alpha_h_same_checkbox.isChecked()
            else None
        )

        self.nmf_rescale_checkbox = qtw.QCheckBox("Rescale")
        self.algorithm_parameters_layout.addWidget(self.nmf_rescale_checkbox, 2,2)
        self.nmf_rescale_checkbox.setToolTip("Rescales scores to sum to 1")


        # Ignored NMF parmeters:
        # random_state
        # verbose
        # shuffle

        self.nmf_algorithm_widgets = [
            self.nmf_init_dropdown, self.nmf_solver_dropdown,
            self.nmf_max_iter_spinbox, self.nmf_tol_spinbox,
            self.nmf_l1_ratio_spinbox, self.nmf_beta_loss_dropdown,
            self.nmf_alpha_w_spinbox, self.nmf_alpha_h_spinbox,
            self.nmf_alpha_h_same_checkbox, self.nmf_rescale_checkbox
        ]

        ##########################
        ##### ICA parameters #####
        ##########################
        self.ica_algorithm_dropdown = qtw.QComboBox()
        self.ica_algorithm_dropdown.addItems(['parallel', 'deflation'])
        self.algorithm_parameters_layout.addWidget(self.ica_algorithm_dropdown, 1,0)
        self.ica_algorithm_dropdown.setToolTip("[algorithm]\n" \
        "Specify which algorithm to use:\n" \
        "parallel is default")

        self.ica_whiten_dropdown = qtw.QComboBox()
        self.ica_whiten_dropdown.addItems(['unit-variance', 'arbitrary-variance', 'False'])
        self.algorithm_parameters_layout.addWidget(self.ica_whiten_dropdown, 1,1)
        self.ica_whiten_dropdown.setToolTip(
            "[whiten]\n"
            "Whitening strategy to use. False means no whitening:\n"
            "unit-variance (default): the whitening matrix is rescaled"
            "to ensure each recovered source has unit variance\n"
            "arbitrary-variance: a whitening with variance arbitrary is used\n"
            "False: Data is considered whitened and no whitening is performed"
        )

        self.ica_max_iter_spinbox = qtw.QSpinBox()
        self.ica_max_iter_spinbox.setMinimum(0)
        self.ica_max_iter_spinbox.setMaximum(999999)
        self.ica_max_iter_spinbox.setValue(2500)
        self.algorithm_parameters_layout.addWidget(self.ica_max_iter_spinbox, 2,0)
        self.ica_max_iter_spinbox.setToolTip("[max_iter]\n" \
        "Maximum number of iterations during fit")

        self.ica_tol_spinbox = gui_helper.SciSpinBox()
        self.ica_tol_spinbox.setMinimum(0)
        self.ica_tol_spinbox.setValue(1e-4)
        self.algorithm_parameters_layout.addWidget(self.ica_tol_spinbox, 2,1)
        self.ica_tol_spinbox.setToolTip(
            "[tol]\n"
            "A positive scalar giving the tolerance"
            "at which the un-mixing matrix is considered to have converged"
        )

        self.ica_fun_dropdown = qtw.QComboBox()
        self.ica_fun_dropdown.addItems(['logcosh', 'exp', 'cube'])
        self.algorithm_parameters_layout.addWidget(self.ica_fun_dropdown, 1,2)
        self.ica_fun_dropdown.setToolTip(
            "[fun]\n"
            "The functional form of the G function used in the approximation to neg-entropy:\n"
            "logcosh is default"
            )

        self.ica_whiten_solver_dropdown = qtw.QComboBox()
        self.ica_whiten_solver_dropdown.addItems(['svd', 'eigh'])
        self.algorithm_parameters_layout.addWidget(self.ica_whiten_solver_dropdown, 2,2)
        self.ica_whiten_solver_dropdown.setToolTip(
            "[whiten_solver]\n"
            "The solver to use for whitening:\n"
            "svd (default): More numerically stable if the problem is degenerate,"
            "and often faster for scattering\n"
            "eigh: More memory efficient when there are more scans than datapoints"
            "(rare in scattering)"
        )

        # Ignored ICA parameters:
        # fun_args
        # w_init
        # random_state

        self.ica_algorithm_widgets = [
            self.ica_algorithm_dropdown,
            self.ica_whiten_dropdown,
            self.ica_max_iter_spinbox,
            self.ica_tol_spinbox,
            self.ica_fun_dropdown,
            self.ica_whiten_solver_dropdown
        ]

        ###########################
        ##### SNMF parameters #####
        ###########################
        self.snmf_min_iter_spinbox = qtw.QSpinBox()
        self.snmf_min_iter_spinbox.setMinimum(0)
        self.snmf_min_iter_spinbox.setMaximum(999999)
        self.snmf_min_iter_spinbox.setValue(20)
        self.algorithm_parameters_layout.addWidget(self.snmf_min_iter_spinbox, 2,0)
        self.snmf_min_iter_spinbox.setToolTip("[min_iter]\n" \
        "Minimum number of iterations before terminating optimzation")

        self.snmf_max_iter_spinbox = qtw.QSpinBox()
        self.snmf_max_iter_spinbox.setMinimum(0)
        self.snmf_max_iter_spinbox.setMaximum(999999)
        self.snmf_max_iter_spinbox.setValue(500)
        self.algorithm_parameters_layout.addWidget(self.snmf_max_iter_spinbox, 2,1)
        self.snmf_max_iter_spinbox.setToolTip("[max_iter]\n" \
        "Maximum number of iterations before terminating optimzation")

        self.snmf_tol_spinbox = gui_helper.SciSpinBox()
        self.snmf_tol_spinbox.setMinimum(0)
        self.snmf_tol_spinbox.setValue(5e-07)
        self.algorithm_parameters_layout.addWidget(self.snmf_tol_spinbox, 2,2)
        self.snmf_tol_spinbox.setToolTip("[tol]\n" \
        "Convergence threshold.\n" \
        "Minimum fractional improvment to allow terminating optimization")

        self.snmf_rho_spinbox = gui_helper.SciSpinBox()
        self.snmf_rho_spinbox.setMinimum(0)
        self.snmf_rho_spinbox.setValue(0)
        self.snmf_rho_spinbox.setDecimals(0)
        #self.SNMFrho_Spinbox.setMaximum(10000000000)
        #self.SNMFrho_Spinbox.setSingleStep(1)
        self.algorithm_parameters_layout.addWidget(self.snmf_rho_spinbox, 3,0)
        self.snmf_rho_spinbox.setToolTip(
                "[rho]\n"
            "Stretching factor // Stretching regularization hyperparameter."
            "Controls stretching pentalty. \n"
            "Typically adjusted in powers of 10.\n"
            "Zero (default) corresponds to no stretching"
            )

        self.snmf_eta_spinbox = qtw.QDoubleSpinBox()
        self.snmf_eta_spinbox.setMinimum(0)
        self.snmf_eta_spinbox.setValue(0)
        self.snmf_eta_spinbox.setDecimals(5)
        #self.SNMFeta_DSpinbox.setMaximum(10)
        #self.SNMFeta_DSpinbox.setSingleStep(0.00005)
        self.algorithm_parameters_layout.addWidget(self.snmf_eta_spinbox, 3,1)
        self.snmf_eta_spinbox.setToolTip(
            "[eta]\n"
            "Sparsity factor. Should be set to zero (default) for non-sparse data such as PDF.\n"
            "Can be used to improve results for sparse data such as XRD,\n"
            "but due to instability should be used only faster selecting the best value for rho.\n"
            "Suggested adjustment is by powers of 2"
        )

        # Ignored SNMF parmeters:
        # random_state
        # verbose
        # show_plots

        self.snmf_algorithm_widgets = [
            self.snmf_min_iter_spinbox,
            self.snmf_max_iter_spinbox,
            self.snmf_tol_spinbox,
            self.snmf_rho_spinbox,
            self.snmf_eta_spinbox
        ]

        #####################################
        ##### Analysis and plot widgets #####
        #####################################

        self.run_analysis_button = qtw.QPushButton("Run analysis")
        self.run_analysis_button.setSizePolicy(
            qtw.QSizePolicy.Policy.MinimumExpanding,
            qtw.QSizePolicy.Policy.Preferred
        )
        self.grid.addWidget(self.run_analysis_button, 2,2)

        self.results_layout = qtw.QGridLayout()
        self.grid.addLayout(self.results_layout, 3,2)

        self.export_results_checkbox = qtw.QCheckBox("Export results")
        self.export_results_checkbox.setToolTip(
            "Exports plot of dataset, and analysis plot and results\n"
            "For the sake of analysis in other programs data is exported in input format"
            "(2θ not converted to Q)"
        )
        self.results_layout.addWidget(self.export_results_checkbox, 0,0)

        self.export_recons_checkbox = qtw.QCheckBox("Reconstructions")
        self.export_recons_checkbox.setToolTip("Exports the reconstructed dataset – scan by scan")
        self.export_recons_checkbox.setDisabled(True)
        self.results_layout.addWidget(self.export_recons_checkbox, 0,1)

        self.export_diffs_checkbox = qtw.QCheckBox("Differences")
        self.export_diffs_checkbox.setToolTip(
            "Exports the difference between dataset and reconstruction – scan by scan"
        )
        self.export_diffs_checkbox.setDisabled(True)
        self.results_layout.addWidget(self.export_diffs_checkbox, 1,1)

        self.export_comp_contribute_checkbox = qtw.QCheckBox("Component contributions")
        self.export_comp_contribute_checkbox.setToolTip(
            "Exports the partial component-wise reconstruction\n"
            "Each component multiplied by its own scoring – scan by scan"
        )
        self.export_comp_contribute_checkbox.setDisabled(True)
        self.results_layout.addWidget(self.export_comp_contribute_checkbox, 1,0)

        self.export_results_checkbox.toggled.connect( # Enable/disable boxes, according to export
            lambda value: (
                self.export_recons_checkbox.setEnabled(value),
                self.export_diffs_checkbox.setEnabled(value),
                self.export_comp_contribute_checkbox.setEnabled(value)
            )
        )
        self.export_results_checkbox.toggled.connect( # If not exporting, uncheck boxes
            lambda value: (
                self.export_recons_checkbox.setChecked(value),
                self.export_diffs_checkbox.setChecked(value),
                self.export_comp_contribute_checkbox.setChecked(value)
            )
            if not value
            else None
        )

        ##############################
        # Reconstructions to display # (REPLACE WITH COMMA-SEPARATED LINEEDIT)
        ##############################
        self.recon_plot_layout = qtw.QGridLayout()
        self.grid.addLayout(self.recon_plot_layout, 4,2)
        self.recon_plot_layout.setAlignment(qtc.Qt.AlignmentFlag.AlignTop)

        self.reconstruct_label = qtw.QLabel("Display Reconstruction Scan #")
        self.reconstruct_label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)
        self.reconstruct_label.setSizePolicy(
            qtw.QSizePolicy.Policy.Minimum,
            qtw.QSizePolicy.Policy.Fixed
        )
        self.recon_plot_layout.addWidget(self.reconstruct_label, 4,0,1,2)

        self.reconstruct_widgets = []

        self.reconstruct_widget0 = qtw.QSpinBox()
        self.reconstruct_widget0.setMinimum(0)
        self.recon_plot_layout.addWidget(self.reconstruct_widget0, 5,0)
        self.reconstruct_widget0.setToolTip(
            "If any are '0',"
            "will reconstruct uniform distribution of scans"
        )
        self.reconstruct_widgets.append(self.reconstruct_widget0)

        self.reconstruct_widget1 = qtw.QSpinBox()
        self.reconstruct_widget1.setMinimum(0)
        self.recon_plot_layout.addWidget(self.reconstruct_widget1, 5,1)
        self.reconstruct_widgets.append(self.reconstruct_widget1)

        self.reconstruct_widget2 = qtw.QSpinBox()
        self.reconstruct_widget2.setMinimum(0)
        self.recon_plot_layout.addWidget(self.reconstruct_widget2, 6,0)
        self.reconstruct_widgets.append(self.reconstruct_widget2)

        self.reconstruct_widget3 = qtw.QSpinBox()
        self.reconstruct_widget3.setMinimum(0)
        self.recon_plot_layout.addWidget(self.reconstruct_widget3, 6,1)
        self.reconstruct_widgets.append(self.reconstruct_widget3)

        self.reconstruct_widget4 = qtw.QSpinBox()
        self.reconstruct_widget4.setMinimum(0)
        self.recon_plot_layout.addWidget(self.reconstruct_widget4, 7,0)
        self.reconstruct_widgets.append(self.reconstruct_widget4)

        self.reconstruct_widget5 = qtw.QSpinBox()
        self.reconstruct_widget5.setMinimum(0)
        self.recon_plot_layout.addWidget(self.reconstruct_widget5, 7,1)
        self.reconstruct_widgets.append(self.reconstruct_widget5)

        self.reconstruct_widget6 = qtw.QSpinBox()
        self.reconstruct_widget6.setMinimum(0)
        self.recon_plot_layout.addWidget(self.reconstruct_widget6, 8,0)
        self.reconstruct_widgets.append(self.reconstruct_widget6)

        self.reconstruct_widget7 = qtw.QSpinBox()
        self.reconstruct_widget7.setMinimum(0)
        self.recon_plot_layout.addWidget(self.reconstruct_widget7, 8,1)
        self.reconstruct_widgets.append(self.reconstruct_widget7)

        self.reconstruct_widget8 = qtw.QSpinBox()
        self.reconstruct_widget8.setMinimum(0)
        self.recon_plot_layout.addWidget(self.reconstruct_widget8, 9,0)
        self.reconstruct_widgets.append(self.reconstruct_widget8)

        self.reconstruct_widget9 = qtw.QSpinBox()
        self.reconstruct_widget9.setMinimum(0)
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
            lambda: qtg.QDesktopServices.openUrl(
                qtc.QUrl.fromLocalFile(f"{self.configpath}/config.dat")
            )
        )
        file_menu.addAction(config_button)

        file_menu.addSeparator()

        self.file_submenu_recent_dirs = file_menu.addMenu("Recent datasets")
        self.file_submenu_recent_dirs_separator = self.file_submenu_recent_dirs.addSeparator()

        self.clear_recent_dirs_button = qtg.QAction("Clear recently opened", self)
        self.clear_recent_dirs_button.triggered.connect(
            lambda: [
                self.file_submenu_recent_dirs.removeAction(action)
                for action in self.file_submenu_recent_dirs.actions()[:-2]
            ]
        )
        self.clear_recent_dirs_button.triggered.connect(self.update_config_file)
        self.file_submenu_recent_dirs.addAction(self.clear_recent_dirs_button)

        self.file_submenu_recent_bkgs = file_menu.addMenu("Recent backgrounds")
        self.file_submenu_recent_bkgs_separator = self.file_submenu_recent_bkgs.addSeparator()

        self.clear_recent_bkgs_button = qtg.QAction("Clear recently opened", self)
        self.clear_recent_bkgs_button.triggered.connect(
            lambda: [
                self.file_submenu_recent_bkgs.removeAction(action)
                for action in self.file_submenu_recent_bkgs.actions()[:-2]
            ]
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
        self.xmin_spinbox.valueChanged.connect(self.update_config_file)
        self.xmax_spinbox.valueChanged.connect(self.update_config_file)
        self.limit_xaxis_checkbox.toggled.connect(self.update_config_file)
        self.scanmin_spinbox.valueChanged.connect(self.update_config_file)
        self.scanmax_spinbox.valueChanged.connect(self.update_config_file)
        self.limit_scans_checkbox.toggled.connect(self.update_config_file)
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

    def closeEvent(self, event): # pylint: disable=invalid-name,unused-argument
        """
        Redefines Qt's closeEvent to shut down the app if main window closes
        """
        qtw.QApplication.quit()

    def about_button_clicked(self):
        """
        Handles the About Window
        """
        if self.about_window is None:
            self.about_window = gui_helper.AboutDialog(self)
            self.about_window.show()
        else:
            self.about_window.close()
            self.about_window = None

    def display_algorithm_widgets(self):
        """
        Shows/hides widgets based on chosen algorithm
        """
        for widget in (self.pca_algorithm_widgets +
                       self.nmf_algorithm_widgets +
                       self.ica_algorithm_widgets +
                       self.snmf_algorithm_widgets):
            widget.hide()
        match self.algorithm_group.checkedButton().text():
            case "PCA":
                self.calc_err_checkbox.setEnabled(True)
                self.pca_whiten_checkbox.show()
                self.pca_solver_dropdown.show()
                if self.pca_solver_dropdown.currentText=='arpack':
                    self.pca_tol_spinbox.show()
                else:
                    self.pca_tol_spinbox.hide()
                if self.pca_solver_dropdown.currentText=='randomized':
                    self.pca_iterated_power_auto_checkbox.show()
                    self.pca_iterated_power_spinbox.show()
                    self.pca_n_oversampled_spinbox.show()
                    self.pca_power_iteration_normalizer_dropdown.show()
                else:
                    self.pca_iterated_power_auto_checkbox.hide()
                    self.pca_iterated_power_spinbox.hide()
                    self.pca_n_oversampled_spinbox.hide()
                    self.pca_power_iteration_normalizer_dropdown.hide()
            case "NMF":
                self.calc_err_checkbox.setEnabled(True)
                for widget in self.nmf_algorithm_widgets:
                    widget.show()
                if self.nmf_solver_dropdown.currentText=='mu':
                    self.nmf_beta_loss_dropdown.show()
                else:
                    self.nmf_beta_loss_dropdown.hide()
                if self.nmf_solver_dropdown.currentText=='cd':
                    self.nmf_l1_ratio_spinbox.show()
                else:
                    self.nmf_l1_ratio_spinbox.show()
            case "ICA":
                self.calc_err_checkbox.setChecked(False)
                self.calc_err_checkbox.setEnabled(False)
                for widget in self.ica_algorithm_widgets:
                    widget.show()
            case "SNMF":
                self.calc_err_checkbox.setChecked(False)
                self.calc_err_checkbox.setEnabled(False)
                for widget in self.snmf_algorithm_widgets:
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
        Sets the dataset directory and gets details for widget behavior
        """
        if not indir: # None from function def, False from "Load directory" widget
            if self.indir_label.text() == "Select the folder containing your data files":
                indir = str(qtw.QFileDialog.getExistingDirectory(self))
            else:
                indir = str(qtw.QFileDialog.getExistingDirectory(
                    self,
                    directory=self.indir_label.text())
                )
            if indir == "": # If the operation was cancelled
                return

        try:
            self.set_widget_limits(indir)
        except (FileNotFoundError, ValueError, IOError) as e:
            qtw.QMessageBox.critical(
                self,
                "Dataset",
                "Could not import dataset.\n\n"
                f"Details:\n{type(e).__name__}: {e}\n\n"
                "Ensure directory and files exist and are in the correct format."
            )
            return

        self.indir_label.setText(indir)
        gui_helper.add_recent(
            indir,
            self.file_submenu_recent_dirs,
            action_func=self.set_indir,
            update_func=self.update_config_file
        )

    def set_bkgfile(self, infile=None):
        """
        Sets the background file, tests its validity and imports it
        """
        print("Importing background")
        if not infile: # Both "None" and "False" pass as False
            if self.bkg_label.text() == "Select a background for subtraction":
                infile,_ = qtw.QFileDialog.getOpenFileName(
                    self,
                    directory=self.indir_label.text()
                ) # Qt handles invalid-directory, so no try-except is needed
            else:
                infile,_ = qtw.QFileDialog.getOpenFileName(
                    self,
                    directory=os.path.dirname(self.bkg_label.text())
                )
            if infile == "": # If the operation was cancelled
                return

        # Confirm bkgfile can be imported by the program
        try:
            file_worker.import_data(infile) # Fast to read, attribute storage not needed
        except (FileNotFoundError, ValueError, IOError) as e:
            qtw.QMessageBox.critical(
                self,
                "Background",
                "Could not import background.\n\n"
                f"Details:\n{type(e).__name__}: {e}\n\n"
                "Ensure file exists and is in the correct format."
            )
        else:
            self.bkg_label.setText(infile)
            gui_helper.add_recent(
                infile,
                self.file_submenu_recent_bkgs,
                action_func=self.set_bkgfile,
                update_func=self.update_config_file
            )
            print("Background imported\n")

    def set_widget_limits(self, indir=None):
        """
        Sets widget limits based on imported data
        """
        if isinstance(indir,str): # If a non-empty string (Workaround for PyQt signal connections)
            xmin, xmax, scanmax = file_worker.get_dataset_details(indir)

            self.dataset_details["xmin"] = xmin
            self.dataset_details["xmax"] = xmax
            self.dataset_details["scanmax"] = scanmax

        if self.convert_to_q_checkbox.isChecked():
            xmin = funcs.theta_to_q(self.dataset_details["xmin"],self.wavelength_widget.value())
            xmax = funcs.theta_to_q(self.dataset_details["xmax"],self.wavelength_widget.value())
        else:
            xmin = self.dataset_details["xmin"]
            xmax = self.dataset_details["xmax"]
        scanmax = self.dataset_details["scanmax"]

        self.xmin_spinbox.setMinimum(xmin)
        self.xmin_spinbox.setMaximum(xmax)
        self.xmax_spinbox.setMinimum(xmin)
        self.xmax_spinbox.setMaximum(xmax)

        self.scanmin_spinbox.setMaximum(scanmax)
        self.scanmax_spinbox.setMaximum(scanmax)
        for widget in self.reconstruct_widgets:
            widget.setMaximum(scanmax)

        if isinstance(indir,str):
            self.xmin_spinbox.setValue(xmin)
            self.xmax_spinbox.setValue(xmax)
            self.scanmin_spinbox.setValue(1)
            self.scanmax_spinbox.setValue(scanmax)

    def read_config_file(self): # pylint: disable=too-many-branches,too-many-statements
        """
        Reads BaSSET config file to set widget values
        """
        with open(self.configfile, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            for line in lines:
                line = line.replace('\n','')
                name, value = line.split(": ")
                if value != "":
                    match name:
                        case "Current file directory":
                            if os.path.exists(value):
                                self.indir_label.setText(value)
                                self.set_indir(value)
                            elif value!="Select the folder containing your data files":
                                print(f"{value} is not a valid directory")
                        case "Recent directories":
                            for indir in reversed(value.split(", ")):
                                if indir != "":
                                    gui_helper.add_recent(
                                        indir,
                                        self.file_submenu_recent_dirs,
                                        action_func=self.set_indir,
                                        update_func=self.update_config_file
                                    )
                        case "Background file":
                            self.bkg_label.setText(value)
                        case "Background scale":
                            self.bkg_scale_spinbox.setValue(float(value))
                        case "Recent backgrounds":
                            for infile in reversed(value.split(", ")):
                                if infile != "":
                                    gui_helper.add_recent(
                                        infile,
                                        self.file_submenu_recent_bkgs,
                                        action_func=self.set_bkgfile,
                                        update_func=self.update_config_file
                                    )
                        case "Input format":
                            for button in self.input_format_group.buttons():
                                if value == button.text():
                                    button.setChecked(True)
                            if value not in (
                                button.text() for button in self.input_format_group.buttons()
                            ):
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
                            self.wavelength_widget.setValue(float(value))
                        case "Limit x-axis":
                            if value=="True":
                                self.limit_xaxis_checkbox.setChecked(True)
                            elif value=="False":
                                self.limit_xaxis_checkbox.setChecked(False)
                                self.xmin_spinbox.setEnabled(False)
                                self.xmax_spinbox.setEnabled(False)
                            else:
                                print("Did not recognize {line} as True/False")
                        case "X-axis range":
                            xmin, xmax = value.split(",")
                            self.xmin_spinbox.setValue(float(xmin))
                            self.xmax_spinbox.setValue(float(xmax))
                        case "Limit scans":
                            if value=="True":
                                self.limit_scans_checkbox.setChecked(True)
                            elif value=="False":
                                self.limit_scans_checkbox.setChecked(False)
                                self.scanmin_spinbox.setEnabled(False)
                                self.scanmax_spinbox.setEnabled(False)
                            else:
                                print("Did not recognize {line} as True/False")
                        case "Scans range":
                            scanmin, scnamax = value.split(",")
                            self.scanmin_spinbox.setValue(int(scanmin))
                            self.scanmax_spinbox.setValue(int(scnamax))
                        case "Algorithm":
                            for button in self.algorithm_group.buttons():
                                if value == button.text():
                                    button.setChecked(True)
                            if value not in (
                                button.text() for button in self.algorithm_group.buttons()
                            ):
                                print(f"Did not recognize \"{value}\" as an algorithm")
                        case "Number of components":
                            self.comp_num_slider.setValue(int(value))
                            self.comp_num_label.setText(value)
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
            outfile.write(f"Current file directory: {
                self.indir_label.text()
                if self.indir_label.text()!='Select the folder containing your data files'
                else ''
            }\n")
            outfile.write(f"Recent directories: {', '.join(
                action.text()
                for action in self.file_submenu_recent_dirs.actions()[:-2]
            )}\n")
            outfile.write(f"Background file: {
                self.bkg_label.text()
                if self.bkg_label.text()!='Select a background for subtraction'
                else ''
            }\n")
            outfile.write(f"Background scale: {self.bkg_scale_spinbox.value():.3f}\n")
            outfile.write(
                "Recent backgrounds: "
                f"{', '.join(
                    action.text()
                    for action in self.file_submenu_recent_bkgs.actions()[:-2]
                )}\n")
            outfile.write(f"Input format: {self.input_format_group.checkedButton().text()}\n")
            outfile.write(f"Convert to Q: {self.convert_to_q_checkbox.isChecked()}\n")
            outfile.write(f"Wavelength: {self.wavelength_widget.value():.6f}\n")
            outfile.write(f"Limit x-axis: {self.limit_xaxis_checkbox.isChecked()}\n")
            outfile.write(
                "X-axis range: "
                f"{self.xmin_spinbox.value():.2f},"
                f"{self.xmax_spinbox.value():.2f}\n"
            )
            outfile.write(f"Limit scans: {self.limit_scans_checkbox.isChecked()}\n")
            outfile.write(
                "Scans range: "
                f"{self.scanmin_spinbox.value()},"
                f"{self.scanmax_spinbox.value()}\n"
            )
            outfile.write(f"Algorithm: {self.algorithm_group.checkedButton().text()}\n")
            outfile.write(f"Number of components: {self.comp_num_slider.value()}\n")
            outfile.write(f"Calculate errors: {self.calc_err_checkbox.isChecked()}\n")

    def plot_dataset(self):
        """
        Plots the input dataset as a waterfall plot
        """
        print("Plotting input dataset\n")

        if self.angles is None or self.intensities is None:
            try:
                self.angles, self.intensities = file_worker.import_dataset(self.indir_label.text())
            except (FileNotFoundError, ValueError, IOError) as e:
                qtw.QMessageBox.critical(
                    self,
                    "Dataset",
                    "Could not import dataset.\n\n"
                    f"Details:\n{type(e).__name__}: {e}\n\n"
                    "Ensure directory and files exist and are in the correct format."
                )
                return

        angles = self.angles.copy()
        intensities = self.intensities.copy()

        if self.convert_to_q_checkbox.isChecked():
            xlabel = self.q_button.text()
            angles = funcs.theta_to_q(angles, self.wavelength_widget.value())
        else:
            xlabel = self.input_format_group.checkedButton().text()

        if self.bkg_label.text() != "Select a background for subtraction":
            try:
                _, bkgintensity = file_worker.import_data(self.bkg_label.text())
            except (FileNotFoundError, ValueError, IOError) as e:
                qtw.QMessageBox.critical(
                    self,
                    "Background",
                    "Could not import background.\n\n"
                    f"Details:\n{type(e).__name__}: {e}\n\n"
                    "Ensure file exists and is in the correct format."
                )
                return
            try:
                intensities -= bkgintensity*self.bkg_scale_spinbox.value()
            except ValueError as e:
                qtw.QMessageBox.critical(
                    self,
                    "Background subtraction",
                    "Could not perform background subtraction.\n\n"
                    f"Details:\n{type(e).__name__}: {e}\n\n"
                    "Ensure dataset and background are equally long or remove background."
                )
                return

        if self.limit_scans_checkbox.isChecked():
            loopmin = self.scanmin_spinbox.value()-2
            loopmax = self.scanmax_spinbox.value()-1
        else:
            loopmin = -1
            loopmax = len(intensities)-1

        fig = plt.figure()
        cmap = plt.get_cmap('inferno')
        colors = cmap(np.linspace(0.8, 0, len(intensities)))
        stagger_factor = np.max(intensities) / (15*len(intensities))
        stagger_max = 0

        for i in range(loopmax, loopmin, -1): # Plots in reverse so last scan is behind
            yaxis = intensities[i]+i*stagger_factor
            plt.plot(angles[i], yaxis, color=colors[i])
            stagger_max = max(stagger_max,np.max(yaxis))

        plt.xlabel(xlabel)
        plt.ylabel("Intensity (staggered) [A.U.]")

        if self.limit_xaxis_checkbox.isChecked():
            plt.xlim(self.xmin_spinbox.value(), self.xmax_spinbox.value())
        else:
            plt.xlim(min(angles[i]), max(angles[i]))

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

        if self.angles is None or self.intensities is None:
            try:
                self.angles, self.intensities = file_worker.import_dataset(self.indir_label.text())
            except (FileNotFoundError, ValueError, IOError) as e:
                qtw.QMessageBox.critical(
                    self,
                    "Dataset",
                    "Could not import dataset.\n\n"
                    f"Details:\n{type(e).__name__}: {e}\n\n"
                    "Ensure directory and files exist and are in the correct format."
                )
                return

        angles = self.angles.copy()
        intensities = self.intensities.copy()

        if self.bkg_label.text() != "Select a background for subtraction":
            try:
                _, bkgintensity = file_worker.import_data(self.bkg_label.text())
            except (FileNotFoundError, ValueError, IOError) as e:
                qtw.QMessageBox.critical(
                    self,
                    "Background",
                    "Could not import background.\n\n"
                    f"Details:\n{type(e).__name__}: {e}\n\n"
                    "Ensure file exists and is in the correct format."
                )
                return
            try:
                intensities -= bkgintensity*self.bkg_scale_spinbox.value()
            except ValueError as e:
                qtw.QMessageBox.critical(
                    self,
                    "Background subtraction",
                    "Could not perform background subtraction.\n\n"
                    f"Details:\n{type(e).__name__}: {e}\n\n"
                    "Ensure dataset and background are equally long or remove background."
                )
                return

        if self.convert_to_q_checkbox.isChecked():
            angles = funcs.theta_to_q(angles, self.wavelength_widget.value())

        if self.limit_xaxis_checkbox.isChecked(): # Crop xrange
            xmin_index = np.searchsorted(angles[0], self.xmin_spinbox.value(), side='left')
            xmax_index = np.searchsorted(angles[0], self.xmax_spinbox.value(), side='right')
            angles = angles[:,xmin_index:xmax_index]
            intensities = intensities[:,xmin_index:xmax_index]
        if self.limit_scans_checkbox.isChecked(): # Crop scans
            scanmin = self.scanmin_spinbox.value()-1
            scanmax = self.scanmax_spinbox.value()-1
            angles = angles[scanmin:scanmax,:]
            intensities = intensities[scanmin:scanmax,:]

        errors = None
        lift_factor = None
        stretch = None
        match self.algorithm_group.checkedButton().text():
            case "PCA":
                fitted, transformed, reconstructed = analysis.PCA_analysis(
                    intensities,
                    comp_num,
                    whiten=self.pca_whiten_checkbox.isChecked(),
                    svd_solver=self.pca_solver_dropdown.currentText(),
                    tol=self.pca_tol_spinbox.value(),
                    iterated_power=(
                        'auto'
                        if self.pca_iterated_power_auto_checkbox.isChecked
                        else self.pca_iterated_power_spinbox.value()
                    ),
                    n_oversamples=self.pca_n_oversampled_spinbox.value(),
                    power_iteration_normalizer=(
                        self.pca_power_iteration_normalizer_dropdown.currentText()
                    )
                )
            case "NMF":
                fitted, transformed, reconstructed, errors, lift_factor = analysis.NMF_analysis(
                    intensities,
                    comp_num,
                    init=self.nmf_init_dropdown.currentText(),
                    solver=self.nmf_solver_dropdown.currentText(),
                    beta_loss=(
                        self.nmf_beta_loss_dropdown.currentText()
                        if self.nmf_solver_dropdown.currentText()=="mu"
                        else "frobenius"
                    ), # Handles 'cd' solver not being compatible with non-frobenius beta_loss
                    tol=self.nmf_tol_spinbox.value(),
                    max_iter=self.nmf_max_iter_spinbox.value(),
                    alpha_W=self.nmf_alpha_w_spinbox.value(),
                    alpha_H=(
                        'same'
                        if self.nmf_alpha_h_same_checkbox.isChecked()
                        else self.nmf_alpha_h_spinbox.value()
                    ),
                    l1_ratio=self.nmf_l1_ratio_spinbox.value(),
                    calc_err=self.calc_err_checkbox.isChecked(),
                    rescale=self.nmf_rescale_checkbox.isChecked(),
                    exp_win={
                        'enable': self.exp_win_enable_checkbox.isChecked(),
                        'do_custom': self.exp_win_custom_checkbox.isChecked(),
                        'num_win': self.exp_win_num_spinbox.value(),
                        'comps': np.fromstring(
                            self.exp_win_comps_line.text(),
                            dtype=int,
                            sep=','
                        ),
                        'win_custom': np.fromstring(
                            self.exp_win_custom_line.text(),
                            dtype=int,
                            sep=','
                        )
                    }
                )
            case "ICA":
                fitted, transformed, reconstructed = analysis.ICA_analysis(
                    intensities,
                    comp_num,
                    algorithm=self.ica_algorithm_dropdown.currentText(),
                    whiten=(
                        False
                        if self.ica_whiten_dropdown.currentText()=='False'
                        else self.ica_whiten_dropdown.currentText()
                    ),
                    fun=self.ica_fun_dropdown.currentText(),
                    max_iter=self.ica_max_iter_spinbox.value(),
                    tol=self.ica_tol_spinbox.value(),
                    whiten_solver=self.ica_whiten_solver_dropdown.currentText(),
                    calc_err=self.calc_err_checkbox.isChecked()
                )
            case "SNMF":
                fitted, transformed, reconstructed, errors, stretch = analysis.SNMF_analysis(
                    intensities,
                    comp_num,
                    min_iter=self.snmf_min_iter_spinbox.value(),
                    max_iter=self.snmf_max_iter_spinbox.value(),
                    tol=self.snmf_tol_spinbox.value(),
                    rho=self.snmf_rho_spinbox.value(),
                    eta=self.snmf_eta_spinbox.value(),
                    calc_err=self.calc_err_checkbox.isChecked()
                )

        print("Analysis completed\n")
        self.plot_analysis(
            comp_num,
            angles,
            intensities,
            fitted,
            transformed,
            reconstructed,
            errors=errors,
            lift_factor=lift_factor,
            stretch=stretch
        )

    def plot_analysis(self, comp_num, angles, intensities, fitted, transformed, reconstructed, *,
                      errors=None,
                      lift_factor=None,
                      stretch=None
    ):
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
            recon_num = list(np.linspace(
                0, len(reconstructed), comp_num-1,
                endpoint=False, dtype=int
            ))
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

            ax_recon.plot(
                angles[recon_num[i]], reconstructed[recon_num[i]],
                color="#DD0000", label="Reconstructed"
            )
            ax_recon.plot(
                angles[recon_num[i]], intensities[recon_num[i]],
                "k:", label="Original"
            )
            ax_recon.plot(
                angles[recon_num[i]], difference-distance*1.05,
                color="#2EC483", label="Difference"
            )
            ax_recon.ticklabel_format(axis="y", style="sci", scilimits=[0,0])
            ax_recon.set_title(f"Scan {recon_num[i]+1}") # Reconstruction num
            ax_recon.set_xlabel(xlabel)
            ax_recon.sharex(ax_comp)
            ax_recon.set_ylim(
                0.05*(np.min(difference)-distance)*1.05 - max(
                    np.max(intensities[recon_num[i]]),
                    np.max(reconstructed[recon_num[i]])
                ),
                1.05*max(
                    np.max(intensities[recon_num[i]]),
                    np.max(reconstructed[recon_num[i]])
                )
            )

            ax_scores.plot(
                np.arange(1,len(angles)+1), transformed[:,i],
                color=colors[i], label=f"Comp. {i+1}"
            )

            if i==0:
                ax_comp_master = ax_comp
            else:
                ax_comp.sharex(ax_comp_master)

        for i, recon in enumerate(recon_num):
            ax_scores.axvline(recon+1, color="k", linestyle=":", label=("Recons" if i==0 else ''))

        # Plot windows if expanding windows are used
        if self.exp_win_enable_checkbox.isChecked():
            if self.exp_win_custom_checkbox.isChecked():
                win_ends = np.fromstring(
                    self.exp_win_custom_line.text(),
                    dtype=int,
                    sep=','
                )
            else:
                win_ends = np.linspace(
                    1,
                    len(intensities),
                    self.exp_win_num_spinbox.value()+1,
                    dtype=int
                )[1:] # Exclude start
            for i, win_end in enumerate(win_ends):
                ax_scores.axvline(win_end, color="k", label=("Windows" if i==0 else ''))

        ax_scores.ticklabel_format(axis="y", style="sci", scilimits=[0,0])
        ax_scores.legend()
        l, h = ax_recon.get_legend_handles_labels()
        ax_errors.legend(l, h, title="↑", frameon=True)
        ax_scores.set_xlabel("Scan #")
        ax_scores.set_xlim(1, len(angles))

        if (
            self.nmf_rescale_checkbox.isChecked()
            and self.algorithm_group.checkedButton().text()=="NMF"
        ):
            ax_scores.set_title("Normalized Scores")
            ax_scores.set_ylim(-0.01, 1.01)
        else:
            ax_scores.set_title("Scores")
            ax_scores.set_ylim(min(0,np.min(transformed)), np.max(transformed))

        match self.algorithm_group.checkedButton().text():
            case "PCA":
                ax_errors.plot(np.arange(1, min(10,*np.shape(intensities))+1),
                               fitted.explained_variance_ratio_*100, "ko--"
                )
                ax_errors.set_title("Explained variances")
            case "NMF":
                if errors is not None:
                    ax_errors.plot(np.arange(1, min(10,*np.shape(intensities))+1), errors, "ko--")
                    ax_errors.set_title("Reconstruction error")
            case "ICA":
                pass
            case "SNMF":
                # pylint: disable=unnecessary-pass,pointless-string-statement
                pass
                """
                ax_errors.plot(np.arange(1, min(10,*np.shape(intensities))+1), errors, "ko--")
                ax_errors.set_title("Reconstruction error (Not Implemented)")
                axs[2][2].plot(np.arange(1, min(10,*np.shape(intensities))+1, stretch, "ko--"))
                axs[2][2].set_title("Stretching")
                """
        ax_errors.set_xlim(0.9, 10.1)

        ax_errors.set_xlabel("# of Components")

        fig.canvas.manager.set_window_title(
            f"{self.algorithm_group.checkedButton().text()} ({comp_num}): "
            f"x:({self.xmin_spinbox.value()},{self.xmax_spinbox.value()})"
        )

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
            self.export_results(
                angles,
                intensities,
                fitted,
                transformed,
                reconstructed,
                fig=fig,
                errors=errors,
                lift_factor=lift_factor,
                stretch=stretch
            )

    def write_summary(self, results_path, errors=None, lift_factor=None):
        """
        Writes summary of analysis
        """
        with open(f"{results_path}/summary.txt", "w", encoding='utf-8') as outfile:
            outfile.write(
                "Summary of "
                f"{self.comp_num_slider.value()} component "
                f"{self.algorithm_group.checkedButton().text()} analysis "
                f"of data in \"...{self.indir_label.text()[-51:]}\"\n\n"
            )
            outfile.write(
                "Data was analyzed from "
                f"{self.xmin_spinbox.value()} to "
                f"{self.xmax_spinbox.value()} "
                f"{self.input_format_group.checkedButton().text()}\n"
            )
            outfile.write("The following algorithm parameters where used:\n")
            match self.algorithm_group.checkedButton().text():
                case "PCA":
                    outfile.write(f"Whiten: {self.pca_whiten_checkbox.isChecked()}\n")
                    outfile.write(f"SVD solver: {self.pca_solver_dropdown.currentText()}\n")
                    if self.pca_solver_dropdown.currentText()=="arpack":
                        outfile.write(f"Tolerance: {self.pca_tol_spinbox.value()}\n")
                    if self.pca_solver_dropdown.currentText()=="randomized":
                        outfile.write("# of iterations (Power method): "f"{
                            self.pca_iterated_power_spinbox()
                            if self.pca_iterated_power_auto_checkbox.isChecked()
                            else 'auto'
                        }\n")
                        outfile.write("Additional vectors to sample data: "
                                      f"{self.pca_n_oversampled_spinbox.value()}\n"
                        )
                        outfile.write(
                            "Power iteration normalizer: "
                            f"{self.pca_power_iteration_normalizer_dropdown.currentText()}\n"
                        )
                        outfile.write(
                            "\nThe PCA explained variance for "
                            f"{self.comp_num_slider.value()} components was: "
                            f"{errors[self.comp_num_slider.value()+1]}\n"
                        )
                case "NMF":
                    outfile.write(
                        "Initialization method: "
                        "{self.nmf_init_dropdown.currentText()}\n"
                        )
                    outfile.write(f"Numerical solver: {self.nmf_solver_dropdown.currentText()}\n")
                    if self.nmf_solver_dropdown.currentText()=='mu':
                        outfile.write(
                            "Beta divergence to minimize: "
                            f"{self.nmf_beta_loss_dropdown.currentText()}\n"
                        )
                    outfile.write(f"Tolerance: {self.nmf_tol_spinbox.value()}\n")
                    outfile.write(
                        "Maximum number of iterations: "
                        f"{self.nmf_max_iter_spinbox.value()}\n"
                    )
                    outfile.write(
                        "Regularization constant for features: "
                        f"{self.nmf_alpha_w_spinbox.value()}\n"
                    )
                    outfile.write(
                        "Regularization constant for samples: "
                        f"{self.nmf_alpha_h_spinbox.value()}\n"
                    ) # Should work even if alpha_Hsame=True
                    outfile.write(
                        "Regularization mixing parameter (0=l2, 1=l1): "
                        f"{self.nmf_l1_ratio_spinbox.value()}\n"
                    )
                    if lift_factor<0:
                        outfile.write(
                            "Due to negative values, "
                            f"your dataset was lifted by {lift_factor} during analysis, "
                            "before subsequent lowering\n"
                        )
                    if self.nmf_rescale_checkbox:
                        outfile.write("Scores were rescaled to sum to 1\n")
                    if errors is not None:
                        outfile.write(
                            "\nThe NMF reconstruction error for "
                            f"{self.comp_num_slider.value()} components was: "
                            f"{errors[self.comp_num_slider.value()+1]:.2f}\n"
                        )
                case "ICA":
                    outfile.write(f"Algorithm: {self.ica_algorithm_dropdown.currentText()}\n")
                    outfile.write(f"Whitening strategy: {self.ica_whiten_dropdown.currentText()}\n")
                    outfile.write(
                        "Whitening solver: "
                        f"{self.ica_whiten_solver_dropdown.currentText()}\n"
                    )
                    outfile.write(
                        "Functional G form function: "
                        f"{self.ica_fun_dropdown.currentText()}\n"
                    )
                    outfile.write(
                        "Maximum number of iterations: "
                        f"{self.ica_max_iter_spinbox.value()}\n"
                    )
                    outfile.write(f"Tolerance: {self.ica_tol_spinbox.value()}\n")
                case "SNMF":
                    outfile.write(
                        "Minimum number of iterations: "
                        f"{self.snmf_min_iter_spinbox.value()}\n"
                    )
                    outfile.write(
                        "Maximum number of iterations: "
                        f"{self.snmf_max_iter_spinbox.value()}\n"
                    )
                    outfile.write(f"Tolerance: {self.snmf_tol_spinbox.value()}\n")
                    outfile.write(f"Stretching factor: {self.snmf_rho_spinbox.value()}\n")
                    outfile.write(f"Sparsity factor: {self.snmf_eta_spinbox.value()}\n")
                    outfile.write(
                        "! Be aware that stretching uses division, "
                        "not multiplication (component / stretch) !"
                    )
        print("Summary written")

    def export_results(self, angles, intensities, fitted, transformed, reconstructed, *,
                       fig=None,
                       errors=None,
                       lift_factor=None,
                       stretch=None
    ):
        """
        Handles exporting of results
        """
        print("Exporting results")
        if not os.path.exists(f"{self.indir_label.text()}/BaSSET_results"):
            os.mkdir(f"{self.indir_label.text()}/BaSSET_results")
            print(f"Results can be found in: {self.indir_label.text()}/BaSSET_results")

        export_time = datetime.now().strftime("%y%m%d-%H%M%S")
        results_path = (
            f"{self.indir_label.text()}/BaSSET_results/"
            f"{export_time}_"
            f"{self.algorithm_group.checkedButton().text()}_"
            f"{self.comp_num_slider.value()}"
        )
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
