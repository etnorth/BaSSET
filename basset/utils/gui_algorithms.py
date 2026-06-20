"""
Handles GUI creation, connection and parameters for BaSSET's algorithms
"""

import PyQt6.QtWidgets as qtw

from basset.utils import (
    gui_helper
)


def init_pca_algorithm_widgets(parent):
    """
    Creates and connects widgets for PCA algorithm parameters

    Parameters
    ----------
    parent: QMainWindow
        Window to anchor widgets to
    """
    parent.pca_whiten_checkbox = qtw.QCheckBox("Whiten")
    parent.algorithm_parameters_layout.addWidget(parent.pca_whiten_checkbox, 1,0)
    parent.pca_whiten_checkbox.setToolTip("[whiten]\n" \
    "Whitening will remove some information from the transformed signal\n" \
    "(the relative variance scales of the components) but can sometime\n" \
    "improve the predictive accuracy of the downstream estimators\n" \
    "by making their data respect some hard-wired assumptions")

    parent.pca_solver_dropdown = qtw.QComboBox()
    parent.pca_solver_dropdown.addItems([
        'auto',
        'full',
        'covariance_eigh',
        'arpack',
        'randomized'
    ])
    parent.algorithm_parameters_layout.addWidget(parent.pca_solver_dropdown, 1,1)
    parent.pca_solver_dropdown.setToolTip("[svd_solver]\n" \
    "The type of Singular Value Decomposition solver to use:\n" \
    "auto (default): Chooses solver based on size of dataset and number of components\n" \
    "full: Runs exact full SVD\n" \
    "covariance_eigh: Precomputes covarience for eigenvalue decompositon.\n" \
    "\tEfficient for many scans of few datapoints (rare for scattering)\n" \
    "arpack: Runs SVD truncated to number of components.\n" \
    "\tRequires fewer components than number of scans\n" \
    "randomized: Runs randomized SVD")

    # Only for 'arpack' solver
    parent.pca_tol_spinbox = gui_helper.SciSpinBox()
    parent.pca_tol_spinbox.setMinimum(0)
    parent.pca_tol_spinbox.setValue(0)
    parent.algorithm_parameters_layout.addWidget(parent.pca_tol_spinbox, 1,2)
    parent.pca_tol_spinbox.setToolTip("[tol]\n" \
    "Tolerance for singular values using 'arpack'")
    parent.pca_solver_dropdown.currentTextChanged.connect(
        lambda currentText: parent.pca_tol_spinbox.show()
        if currentText=='arpack'
        else parent.pca_tol_spinbox.hide()
    )

    # Only for 'randomized' solver
    parent.pca_iterated_power_spinbox = qtw.QSpinBox()
    parent.pca_iterated_power_spinbox.setMinimum(0)
    parent.pca_iterated_power_spinbox.setMaximum(999999)
    parent.pca_iterated_power_spinbox.setValue(0)
    parent.pca_iterated_power_spinbox.setSingleStep(100)
    parent.algorithm_parameters_layout.addWidget(parent.pca_iterated_power_spinbox, 2,0)
    parent.pca_iterated_power_spinbox.setToolTip("[iterated_power]\n" \
    "Number of iterations for the power method in 'randomized'")

    # Only for 'randomized' solver
    parent.pca_iterated_power_auto_checkbox = qtw.QCheckBox("auto")
    parent.algorithm_parameters_layout.addWidget(parent.pca_iterated_power_auto_checkbox, 2,1)
    parent.pca_iterated_power_auto_checkbox.setToolTip("[iterated_power]\n" \
    "Automatically choose number of iterations")
    parent.pca_iterated_power_auto_checkbox.clicked.connect(
        lambda state: parent.pca_iterated_power_spinbox.setDisabled(True)
        if state
        else parent.pca_iterated_power_spinbox.setDisabled(False)
    )

    # Only for 'randomized' solver
    parent.pca_n_oversampled_spinbox = qtw.QSpinBox()
    parent.pca_n_oversampled_spinbox.setMinimum(0)
    parent.pca_n_oversampled_spinbox.setMaximum(50)
    parent.pca_n_oversampled_spinbox.setValue(10)
    parent.pca_n_oversampled_spinbox.setSingleStep(1)
    parent.algorithm_parameters_layout.addWidget(parent.pca_n_oversampled_spinbox, 2,2)
    parent.pca_n_oversampled_spinbox.setToolTip("[n_oversamples]\n" \
    "Additional number of random vectors to sample using 'randomized'")

    # Only for 'randomized' solver // Not for 'arpack' solver
    parent.pca_power_iteration_normalizer_dropdown = qtw.QComboBox()
    parent.pca_power_iteration_normalizer_dropdown.addItems(['auto', 'QR', 'LU', 'none'])
    parent.algorithm_parameters_layout.addWidget(parent.pca_power_iteration_normalizer_dropdown,1,2)
    parent.pca_power_iteration_normalizer_dropdown.setToolTip("[power_iteration_normalizer]\n" \
    "Power iteration normalizer using 'randomized'")
    parent.pca_solver_dropdown.currentTextChanged.connect(
        lambda currentText: (
            parent.pca_iterated_power_spinbox.show(),
            parent.pca_iterated_power_auto_checkbox.show(),
            parent.pca_n_oversampled_spinbox.show(),
            parent.pca_power_iteration_normalizer_dropdown.show()
        )
        if currentText=='randomized'
        else (
            parent.pca_iterated_power_spinbox.hide(),
            parent.pca_iterated_power_auto_checkbox.hide(),
            parent.pca_n_oversampled_spinbox.hide(),
            parent.pca_power_iteration_normalizer_dropdown.hide()
        )
    )

    # Ignored PCA parameters:
    # random_state

    parent.pca_algorithm_widgets = [
        parent.pca_whiten_checkbox, parent.pca_solver_dropdown,
        parent.pca_tol_spinbox, parent.pca_iterated_power_spinbox,
        parent.pca_iterated_power_auto_checkbox, parent.pca_n_oversampled_spinbox,
        parent.pca_n_oversampled_spinbox, parent.pca_power_iteration_normalizer_dropdown
    ]

def init_nmf_algorithm_widgets(parent):
    """
    Creates and connects widgets for NMF algorithm parameters

    Parameters
    ----------
    parent: QMainWindow
        Window to anchor widgets to
    """
    parent.nmf_init_dropdown = qtw.QComboBox()
    parent.nmf_init_dropdown.addItems(["nndsvda", "random", "nndsvd", "nndsvdar"])
    parent.algorithm_parameters_layout.addWidget(parent.nmf_init_dropdown, 1,0)
    parent.nmf_init_dropdown.setToolTip("[init]\n" \
    "Method used to initialize the procedure:\n" \
    "('nndsvda' is recommended for PDF and 'nndsvd' is recommended for XRD)\n" \
    "nndsvda (default): Better when sparsity is not desired (PDF)\n" \
    "random: Random non-negative matrices\n" \
    "nndsvd: Non-negative Double Singular Value Decomposition (better for sparseness) (XRD)\n" \
    "nndsvdar: Faster, less accurate alternative to NNDSVDa when sparsity is not desired")

    parent.nmf_solver_dropdown = qtw.QComboBox()
    parent.nmf_solver_dropdown.addItems(["cd", "mu"])
    parent.algorithm_parameters_layout.addWidget(parent.nmf_solver_dropdown, 1,1)
    parent.nmf_solver_dropdown.setToolTip("[solver]\n" \
    "Numerical solver to use:\n" \
    "cd (default): Coordinate Descent\n" \
    "mu: Multiplicative Update\n" \
    "('mu' gives poor results with 'nndsvd' as it cannot update zeros in initialization)")

    parent.nmf_max_iter_spinbox = qtw.QSpinBox()
    parent.nmf_max_iter_spinbox.setMinimum(1)
    parent.nmf_max_iter_spinbox.setMaximum(999999)
    parent.nmf_max_iter_spinbox.setValue(2500)
    parent.algorithm_parameters_layout.addWidget(parent.nmf_max_iter_spinbox, 2,0)
    parent.nmf_max_iter_spinbox.setToolTip("[max_iter]\n" \
    "Maximum number of iterations before timing out")

    parent.nmf_tol_spinbox = gui_helper.SciSpinBox()
    parent.nmf_tol_spinbox.setMinimum(0)
    parent.nmf_tol_spinbox.setValue(1e-4)
    parent.algorithm_parameters_layout.addWidget(parent.nmf_tol_spinbox, 2,1)
    parent.nmf_tol_spinbox.setToolTip("[tol]\n" \
    "Tolerance of the stopping condition")

    # Used in 'cd' solver
    parent.nmf_l1_ratio_spinbox = qtw.QDoubleSpinBox()
    parent.nmf_l1_ratio_spinbox.setMinimum(0)
    parent.nmf_l1_ratio_spinbox.setMaximum(1)
    parent.nmf_l1_ratio_spinbox.setValue(0)
    parent.nmf_l1_ratio_spinbox.setSingleStep(0.05)
    parent.algorithm_parameters_layout.addWidget(parent.nmf_l1_ratio_spinbox, 1,2)
    parent.nmf_l1_ratio_spinbox.setToolTip("[l1_ratio]\n" \
    "Regularization mixing parameter:\n" \
    "(0, default): elementwise L2 penalty aka. Frobenius Norm\n"
    "(1): elementwwise L1 penalty (better for sparseness)\n")
    parent.nmf_solver_dropdown.currentTextChanged.connect(
        lambda currentText: parent.nmf_l1_ratio_spinbox.show()
        if currentText=='cd'
        else parent.nmf_l1_ratio_spinbox.hide()
    )

    # Only for 'mu' solver
    parent.nmf_beta_loss_dropdown = qtw.QComboBox()
    # "itakura-saito" not included. Cannot have zeros, which XRD/PDF often has
    parent.nmf_beta_loss_dropdown.addItems(["frobenius", "kullback-leibler"])
    parent.algorithm_parameters_layout.addWidget(parent.nmf_beta_loss_dropdown, 1,2)
    parent.nmf_beta_loss_dropdown.setToolTip("[beta_loss]\n" \
    "Beta divergence to be minimized," \
    "measuring the distance between X and the dot product WH using 'mu':\n" \
    "frobenius is default")
    parent.nmf_solver_dropdown.currentTextChanged.connect(
        lambda currentText: parent.nmf_beta_loss_dropdown.show()
        if currentText=='mu'
        else parent.nmf_beta_loss_dropdown.hide()
    )

    parent.nmf_alpha_w_spinbox = qtw.QDoubleSpinBox()
    parent.nmf_alpha_w_spinbox.setMinimum(0)
    parent.nmf_alpha_w_spinbox.setMaximum(10)
    parent.nmf_alpha_w_spinbox.setValue(0)
    parent.nmf_alpha_w_spinbox.setDecimals(5)
    parent.nmf_alpha_w_spinbox.setSingleStep(0.00005)
    parent.algorithm_parameters_layout.addWidget(parent.nmf_alpha_w_spinbox, 3,0)
    parent.nmf_alpha_w_spinbox.setToolTip(
        "[alpha_W]\n"
        "Constant that multiplies the regularization terms of the mixing of features:\n"
        "(Regularization is a penalty term to constrain parameter"
        "complexity and reduce overfitting)\n"
        "0 (default) means no regularization"
    )

    parent.nmf_alpha_h_spinbox = qtw.QDoubleSpinBox()
    parent.nmf_alpha_h_spinbox.setMinimum(0)
    parent.nmf_alpha_h_spinbox.setMaximum(10)
    parent.nmf_alpha_h_spinbox.setValue(0)
    parent.nmf_alpha_h_spinbox.setDecimals(5)
    parent.nmf_alpha_h_spinbox.setSingleStep(0.00005)
    parent.nmf_alpha_h_spinbox.setDisabled(True)
    parent.algorithm_parameters_layout.addWidget(parent.nmf_alpha_h_spinbox, 3,1)
    parent.nmf_alpha_h_spinbox.setToolTip("[alpha_H]\n" \
    "Constant that multiplies the regularization terms of the features:\n" \
    "0 means no regularization. By default same as alpha_W ")

    parent.nmf_alpha_h_same_checkbox = qtw.QCheckBox("same")
    parent.nmf_alpha_h_same_checkbox.setChecked(True)
    parent.algorithm_parameters_layout.addWidget(parent.nmf_alpha_h_same_checkbox, 3,2)
    parent.nmf_alpha_h_same_checkbox.setToolTip("[alpha_H]\n" \
    "Use same regularization constant for features and mixing")
    parent.nmf_alpha_h_same_checkbox.clicked.connect(
        lambda state: (
            parent.nmf_alpha_h_spinbox.setDisabled(True),
            parent.nmf_alpha_h_spinbox.setValue(parent.nmf_alpha_w_spinbox.value())
        ) if state
        else parent.nmf_alpha_h_spinbox.setDisabled(False)
    )
    parent.nmf_alpha_w_spinbox.valueChanged.connect(
        lambda value: parent.nmf_alpha_h_spinbox.setValue(value)
        if parent.nmf_alpha_h_same_checkbox.isChecked()
        else None
    )

    parent.nmf_rescale_checkbox = qtw.QCheckBox("Rescale")
    parent.algorithm_parameters_layout.addWidget(parent.nmf_rescale_checkbox, 2,2)
    parent.nmf_rescale_checkbox.setToolTip("Rescales scores to sum to 1")

    # Ignored NMF parmeters:
    # random_state
    # shuffle

    parent.nmf_algorithm_widgets = [
        parent.nmf_init_dropdown, parent.nmf_solver_dropdown,
        parent.nmf_max_iter_spinbox, parent.nmf_tol_spinbox,
        parent.nmf_l1_ratio_spinbox, parent.nmf_beta_loss_dropdown,
        parent.nmf_alpha_w_spinbox, parent.nmf_alpha_h_spinbox,
        parent.nmf_alpha_h_same_checkbox, parent.nmf_rescale_checkbox
    ]

def init_ica_algorithm_widgets(parent):
    """
    Creates and connects widgets for ICA algorithm parameters

    Parameters
    ----------
    parent: QMainWindow
        Window to anchor widgets to
    """
    parent.ica_algorithm_dropdown = qtw.QComboBox()
    parent.ica_algorithm_dropdown.addItems(['parallel', 'deflation'])
    parent.algorithm_parameters_layout.addWidget(parent.ica_algorithm_dropdown, 1,0)
    parent.ica_algorithm_dropdown.setToolTip("[algorithm]\n" \
    "Specify which algorithm to use:\n" \
    "parallel is default")

    parent.ica_whiten_dropdown = qtw.QComboBox()
    parent.ica_whiten_dropdown.addItems(['unit-variance', 'arbitrary-variance', 'False'])
    parent.algorithm_parameters_layout.addWidget(parent.ica_whiten_dropdown, 1,1)
    parent.ica_whiten_dropdown.setToolTip(
        "[whiten]\n"
        "Whitening strategy to use. False means no whitening:\n"
        "unit-variance (default): the whitening matrix is rescaled"
        "to ensure each recovered source has unit variance\n"
        "arbitrary-variance: a whitening with variance arbitrary is used\n"
        "False: Data is considered whitened and no whitening is performed"
    )

    parent.ica_max_iter_spinbox = qtw.QSpinBox()
    parent.ica_max_iter_spinbox.setMinimum(1)
    parent.ica_max_iter_spinbox.setMaximum(999999)
    parent.ica_max_iter_spinbox.setValue(2500)
    parent.algorithm_parameters_layout.addWidget(parent.ica_max_iter_spinbox, 2,0)
    parent.ica_max_iter_spinbox.setToolTip("[max_iter]\n" \
    "Maximum number of iterations during fit")

    parent.ica_tol_spinbox = gui_helper.SciSpinBox()
    parent.ica_tol_spinbox.setMinimum(0)
    parent.ica_tol_spinbox.setValue(1e-4)
    parent.algorithm_parameters_layout.addWidget(parent.ica_tol_spinbox, 2,1)
    parent.ica_tol_spinbox.setToolTip(
        "[tol]\n"
        "A positive scalar giving the tolerance"
        "at which the un-mixing matrix is considered to have converged"
    )

    parent.ica_fun_dropdown = qtw.QComboBox()
    parent.ica_fun_dropdown.addItems(['logcosh', 'exp', 'cube'])
    parent.algorithm_parameters_layout.addWidget(parent.ica_fun_dropdown, 1,2)
    parent.ica_fun_dropdown.setToolTip(
        "[fun]\n"
        "The functional form of the G function used in the approximation to neg-entropy:\n"
        "logcosh is default"
        )

    parent.ica_whiten_solver_dropdown = qtw.QComboBox()
    parent.ica_whiten_solver_dropdown.addItems(['svd', 'eigh'])
    parent.algorithm_parameters_layout.addWidget(parent.ica_whiten_solver_dropdown, 2,2)
    parent.ica_whiten_solver_dropdown.setToolTip(
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

    parent.ica_algorithm_widgets = [
        parent.ica_algorithm_dropdown,
        parent.ica_whiten_dropdown,
        parent.ica_max_iter_spinbox,
        parent.ica_tol_spinbox,
        parent.ica_fun_dropdown,
        parent.ica_whiten_solver_dropdown
    ]

def init_snmf_algorithm_widgets(parent):
    """
    Creates and connects widgets for ICA algorithm parameters

    Parameters
    ----------
    parent: QMainWindow
        Window to anchor widgets to
    """
    parent.snmf_min_iter_spinbox = qtw.QSpinBox()
    parent.snmf_min_iter_spinbox.setMinimum(0)
    parent.snmf_min_iter_spinbox.setMaximum(999999)
    parent.snmf_min_iter_spinbox.setValue(20)
    parent.algorithm_parameters_layout.addWidget(parent.snmf_min_iter_spinbox, 2,0)
    parent.snmf_min_iter_spinbox.setToolTip("[min_iter]\n" \
    "Minimum number of iterations before terminating optimzation")

    parent.snmf_max_iter_spinbox = qtw.QSpinBox()
    parent.snmf_max_iter_spinbox.setMinimum(1)
    parent.snmf_max_iter_spinbox.setMaximum(999999)
    parent.snmf_max_iter_spinbox.setValue(500)
    parent.algorithm_parameters_layout.addWidget(parent.snmf_max_iter_spinbox, 2,1)
    parent.snmf_max_iter_spinbox.setToolTip("[max_iter]\n" \
    "Maximum number of iterations before terminating optimzation")

    parent.snmf_tol_spinbox = gui_helper.SciSpinBox()
    parent.snmf_tol_spinbox.setMinimum(0)
    parent.snmf_tol_spinbox.setValue(5e-07)
    parent.algorithm_parameters_layout.addWidget(parent.snmf_tol_spinbox, 2,2)
    parent.snmf_tol_spinbox.setToolTip("[tol]\n" \
    "Convergence threshold.\n" \
    "Minimum fractional improvment to allow terminating optimization")

    parent.snmf_rho_spinbox = gui_helper.SciSpinBox()
    parent.snmf_rho_spinbox.setMinimum(0)
    parent.snmf_rho_spinbox.setValue(0)
    parent.snmf_rho_spinbox.setDecimals(0)
    parent.algorithm_parameters_layout.addWidget(parent.snmf_rho_spinbox, 3,0)
    parent.snmf_rho_spinbox.setToolTip(
            "[rho]\n"
        "Stretching factor // Stretching regularization hyperparameter."
        "Controls stretching pentalty. \n"
        "Typically adjusted in powers of 10.\n"
        "Zero (default) corresponds to no stretching"
        )

    parent.snmf_eta_spinbox = qtw.QDoubleSpinBox()
    parent.snmf_eta_spinbox.setMinimum(0)
    parent.snmf_eta_spinbox.setValue(0)
    parent.snmf_eta_spinbox.setDecimals(5)
    parent.algorithm_parameters_layout.addWidget(parent.snmf_eta_spinbox, 3,1)
    parent.snmf_eta_spinbox.setToolTip(
        "[eta]\n"
        "Sparsity factor. Should be set to zero (default) for non-sparse data such as PDF.\n"
        "Can be used to improve results for sparse data such as XRD,\n"
        "but due to instability should be used only faster selecting the best value for rho.\n"
        "Suggested adjustment is by powers of 2"
    )

    # Ignored SNMF parmeters:
    # random_state
    # show_plots

    parent.snmf_algorithm_widgets = [
        parent.snmf_min_iter_spinbox,
        parent.snmf_max_iter_spinbox,
        parent.snmf_tol_spinbox,
        parent.snmf_rho_spinbox,
        parent.snmf_eta_spinbox
    ]

def display_algorithm_widgets(parent):
    """
    Shows/hides widgets based on chosen algorithm

    Parameters
    ----------
    parent: QMainWindow
        Window to anchor widgets to
    """
    for widget in (
        parent.pca_algorithm_widgets +
        parent.nmf_algorithm_widgets +
        parent.ica_algorithm_widgets +
        parent.snmf_algorithm_widgets
    ):
        widget.hide()

    parent.calc_err_checkbox.setChecked(False)
    parent.calc_err_checkbox.hide()
    parent.verbose_checkbox.setChecked(False)
    parent.verbose_checkbox.hide()
    parent.exp_win_checkbox.setChecked(False)
    parent.exp_win_checkbox.hide()
    parent.init_guess_checkbox.setChecked(False)
    parent.init_guess_checkbox.hide()

    match parent.algorithm_group.checkedButton().text():
        case "PCA":
            parent.pca_whiten_checkbox.show()
            parent.pca_solver_dropdown.show()
            if parent.pca_solver_dropdown.currentText()=='arpack':
                parent.pca_tol_spinbox.show()
            elif parent.pca_solver_dropdown.currentText()=='randomized':
                parent.pca_iterated_power_auto_checkbox.show()
                parent.pca_iterated_power_spinbox.show()
                parent.pca_n_oversampled_spinbox.show()
                parent.pca_power_iteration_normalizer_dropdown.show()
        case "NMF":
            parent.calc_err_checkbox.show()
            parent.verbose_checkbox.show()
            parent.exp_win_checkbox.show()
            parent.init_guess_checkbox.show()
            for widget in parent.nmf_algorithm_widgets:
                widget.show()
            if parent.nmf_solver_dropdown.currentText()=='mu':
                parent.nmf_l1_ratio_spinbox.hide()
                parent.nmf_beta_loss_dropdown.show()
            elif parent.nmf_solver_dropdown.currentText()=='cd':
                parent.nmf_beta_loss_dropdown.hide()
                parent.nmf_l1_ratio_spinbox.show()
        case "ICA":
            for widget in parent.ica_algorithm_widgets:
                widget.show()
        case "SNMF":
            parent.calc_err_checkbox.show()
            parent.verbose_checkbox.show()
            for widget in parent.snmf_algorithm_widgets:
                widget.show()
