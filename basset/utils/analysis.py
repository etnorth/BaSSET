"""
Module containing algorithm calls for BaSSET's data analysis
"""
# pylint: disable=invalid-name
import sys
import logging

import numpy as np
import torch
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.decomposition._nmf import _initialize_nmf
from diffpy.stretched_nmf.snmf_class import SNMFOptimizer, _reconstruct_matrix
from constrainedmf.nmf.models import NMF as CNMF
from pymcr.mcr import McrAR

class ComponentsResult:
    """
    Tiny Class to make CNMF interfacing with BaSSET easier
    """
    def __init__(self, components_):
        self.components_ = components_


def PCA_analysis(intensities, comp_num, *, # pylint: disable=unused-argument
    whiten,
    svd_solver,
    tol,
    iterated_power,
    n_oversamples,
    power_iteration_normalizer
):
    """
    Calls on sklearn's PCA algorithm and performs a fit to the user's dataset
    """
    n_components = min(10,*np.shape(intensities)) # Uses 10 for reporting explained variances
    pca_model = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver=svd_solver,
        tol=tol,
        iterated_power=iterated_power,
        n_oversamples=n_oversamples,
        power_iteration_normalizer=power_iteration_normalizer
    )
    X = pca_model.fit(intensities)
    transformed = pca_model.transform(intensities)
    reconstructed = pca_model.inverse_transform(transformed)

    return X, transformed, reconstructed

def NMF_analysis(intensities, comp_num, *,
    init,
    solver,
    beta_loss,
    tol,
    max_iter,
    alpha_W,
    alpha_H,
    l1_ratio,
    calc_err,
    exp_win,
    W,
    H,
    verbose=0
):
    """
    Calls on sklearn's NMF algorithm and performs a fit to the user's dataset
    Calculates error for 1-10 components, and if data contains negative values, lifts them
    """
    errors = None

    lift_factor = intensities.min()
    if lift_factor < 0:
        print("\tNegative value found in dataset. Lifting data")
        intensities -= lift_factor

    itakura_saito_lift = 0
    if beta_loss=="itakura-saito" and 0 in intensities:
        print(
            "\tItakura-Saito was selected as solver, but may diverge if data contains zeros. "
            "Adding small value"
        )
        itakura_saito_lift = 1e-2
        intensities += itakura_saito_lift

    if (W is not None) ^ (H is not None): # XOR, if both are none, go on, if one is none do this
        print("\tPerforming NMF with initial guesses as warm-start")
        # Expand W and H to appropriate size
        W_expand, H_expand = _initialize_nmf( # Create full-sized W and H for expanding
            intensities,
            comp_num,
            init=init
        )

        # intensities.shape = (samples, features)
        # W_full.shape = (samples, comp_num)
        # H_full.shape = (comp_num, features)

        if W is None:
            W = W_expand
        else:
            if W.shape[0] < intensities.shape[0]: # Expand samples in guessed scores
                W_vcrop = W_expand[W.shape[0]:,:]
                W = np.vstack([W, W_vcrop])
            if W.shape[1] < comp_num: # Expand components in guessed scores
                W_hcrop = W_expand[:,W.shape[1]:]
                W = np.hstack([W, W_hcrop])

        if H is None:
            H = H_expand
        else:
            if lift_factor < 0:
                H -= lift_factor
            if H.shape[0] < comp_num: # Expand components in guessed components
                H_vcrop = H_expand[H.shape[0]:,:]
                H = np.vstack([H, H_vcrop])
            # H.shape[1] vs. intensities.shape[1] comparison performed pre function call

        nmf_model = NMF( # initialize with 'custom' for W and H guesses
            n_components=comp_num,
            init='custom',
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose
        )
        nmf_model = nmf_model.fit(intensities, W=W, H=H)
        transformed = nmf_model.transform(intensities)
        reconstructed = nmf_model.inverse_transform(transformed)
        print(
            f"\tNMF ({comp_num}) reconstruction error: "
            f"{nmf_model.reconstruction_err_:.6e} in {nmf_model.n_iter_} iterations"
        )

    elif calc_err:
        print("\tPerforming NMF from 2-10 components to calculate error")
        n_components_list = np.arange(1, min(10,*np.shape(intensities))+1, dtype=int)
        errors = np.empty(len(n_components_list))

        for i, n_components in enumerate(n_components_list):
            print(f"Calculating NMF reconstruction error for {n_components} components")
            nmf_model = NMF(
                n_components=n_components,
                init=init,
                solver=solver,
                beta_loss=beta_loss,
                tol=tol,
                max_iter=max_iter,
                alpha_W=alpha_W,
                alpha_H=alpha_H,
                l1_ratio=l1_ratio,
                verbose=verbose
            )
            nmf_model_temp = nmf_model.fit(intensities)
            errors[i] = nmf_model_temp.reconstruction_err_
            print(
                f"\tNMF ({n_components}) reconstruction error: "
                f"{nmf_model_temp.reconstruction_err_:.6e} in {nmf_model.n_iter_} iterations"
            )

            if comp_num == n_components:
                nmf_model = nmf_model_temp
                transformed = nmf_model.transform(intensities)
                reconstructed = nmf_model.inverse_transform(transformed)

    elif exp_win['enable']:
        print("\tPerforming NMF with expanded window fitting")
        if exp_win['do_custom']:
            if exp_win['win_custom'][-1] < len(intensities): # Makes sure it includes entire set
                exp_win['win_custom'] = np.hstack([exp_win['win_custom'],(len(intensities)+1)])
                exp_win['comps'] = np.hstack([exp_win['comps'], comp_num]) #Assume win.len==comp.len
            win_ends = exp_win['win_custom']
        else: # Creates num_win windows with uniform distribution
            win_ends = np.linspace(
                1,
                len(intensities),
                exp_win['num_win']+1,
                dtype=int
            )[1:] # Exclude start
        if exp_win['comps'].size == 0: # If empty, use all from start
            win_comps = np.full(len(win_ends),comp_num)
        else:
            win_comps = exp_win['comps']

        # 'first' iteration outside loop, warm-start inside loop
        nmf_model = NMF(
            n_components=win_comps[0],
            init=init,
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose
        )
        nmf_model = nmf_model.fit(intensities[:win_ends[0],:])
        W = nmf_model.transform(intensities[:win_ends[0],:])
        H = nmf_model.components_
        print(
            f"\tNMF ({win_comps[0]}) [start to {win_ends[0]}] reconstruction error: "
            f"{nmf_model.reconstruction_err_:.6e} in {nmf_model.n_iter_} iterations"
        )

        for i, (win_end, win_comp) in enumerate(zip(win_ends[1:], win_comps[1:])):
            win_intensities = intensities[:win_end,:]
            W_rows_old, prev_comp = W.shape

            W_init, H_init = _initialize_nmf( # Create scores and components for larger window
                win_intensities,
                win_comp,
                init=init
            )

            W_vcrop = W_init[W_rows_old:,:prev_comp] # Get new rows for old components for vstack
            W_hcrop = W_init[:,prev_comp:] # Get new cols for hstack
            W = np.vstack([W, W_vcrop])
            W = np.hstack([W, W_hcrop])

            H_vcrop = H_init[prev_comp:,:] # Get new components (if win_comp=prev_comp stacks None)
            H = np.vstack([H, H_vcrop])

            nmf_model = NMF( # initialize with 'custom' for W and H guesses
                n_components=win_comp,
                init='custom',
                solver=solver,
                beta_loss=beta_loss,
                tol=tol,
                max_iter=max_iter,
                alpha_W=alpha_W,
                alpha_H=alpha_H,
                l1_ratio=l1_ratio,
                verbose=verbose
            )
            nmf_model = nmf_model.fit(win_intensities, W=W, H=H)
            W = nmf_model.transform(win_intensities)
            H = nmf_model.components_
            print(
                f"\tNMF ({win_comp}) [start to {win_end}] reconstruction error: "
                f"{nmf_model.reconstruction_err_:.2} in {nmf_model.n_iter_} iterations"
            )

        transformed = W
        reconstructed = nmf_model.inverse_transform(transformed)

    else:
        nmf_model = NMF(
            n_components=comp_num,
            init=init,
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose
        )
        nmf_model = nmf_model.fit(intensities)
        transformed = nmf_model.transform(intensities)
        reconstructed = nmf_model.inverse_transform(transformed)
        print(
            f"\tNMF ({comp_num}) reconstruction error: "
            f"{nmf_model.reconstruction_err_:.6e} in {nmf_model.n_iter_} iterations"
        )

    if lift_factor < 0: # Lower data below zero again
        intensities += lift_factor
        nmf_model.components_ += lift_factor
        reconstructed += lift_factor

    if itakura_saito_lift != 0:
        intensities -= itakura_saito_lift
        nmf_model.components_ -= itakura_saito_lift
        reconstructed -= itakura_saito_lift

    return nmf_model, transformed, reconstructed, errors, lift_factor

def ICA_analysis(intensities, comp_num, *,
    algorithm,
    whiten,
    fun,
    max_iter,
    tol,
    whiten_solver
):
    """
    Calls on sklearn's ICA algorithm and performs a fit to the user's dataset
    """
    ica_model = FastICA(
        n_components=comp_num,
        algorithm=algorithm,
        whiten=whiten,
        fun=fun,
        max_iter=max_iter,
        tol=tol,
        whiten_solver=whiten_solver
    )
    X = ica_model.fit(intensities)
    transformed = ica_model.transform(intensities)
    reconstructed = ica_model.inverse_transform(transformed)

    return X, transformed, reconstructed

def SNMF_analysis(intensities, comp_num, *,
    min_iter,
    max_iter,
    tol,
    rho,
    eta,
    calc_err,
    verbose=False
):
    """
    Calls on diffpy's stretched-nmf and performs a fit to the user's dataset
    """
    intensities = intensities.T # SNMF: (n_features,n_samples), sklearn: (n_samples,n_features)

    lift_factor = intensities.min()
    if lift_factor < 0:
        intensities -= lift_factor
        print("\tNegative value found in dataset. Lifting data above zero")

    if calc_err:
        n_components_list = np.arange(1, min(10,*np.shape(intensities))+1, dtype=int)
        errors = np.empty(len(n_components_list))
        for i, n_components in enumerate(n_components_list):
            print(f"Calculating SNMF reconstruction error for {n_components} components")
            snmf_model = SNMFOptimizer(
                n_components=n_components,
                min_iter=min_iter,
                max_iter=max_iter,
                tol=tol,
                rho=rho,
                eta=eta,
                show_plots=True,
                verbose=verbose
            )
            X_temp = snmf_model.fit(intensities)
            errors[i] = X_temp.reconstruction_err_
            print(
                f"\tSNMF ({n_components}) reconstruction error: "
                f"{X_temp.reconstruction_err_:.6e}"
            )
            if comp_num == n_components:
                X = X_temp
                transformed = snmf_model.weights_
                stretch = snmf_model.stretch_
                reconstructed = _reconstruct_matrix(snmf_model.components_, transformed, stretch)
    else:
        snmf_model = SNMFOptimizer(
            n_components=comp_num,
            min_iter=min_iter,
            max_iter=max_iter,
            tol=tol,
            rho=rho,
            eta=eta,
            show_plots=True,
            verbose=verbose
        )
        errors = None
        X = snmf_model.fit(intensities)
        transformed = snmf_model.weights_
        stretch = snmf_model.stretch_
        reconstructed = _reconstruct_matrix(snmf_model.components_, transformed, stretch)
        print(f"\tNMF ({comp_num}) reconstruction error: {X.reconstruction_err_:.6e}")

    if lift_factor < 0: # Lower data below zero again
        intensities += lift_factor
        snmf_model.components_ += lift_factor
        reconstructed += lift_factor

    # Transpose to match sklearn
    X.components_ = X.components_.T
    transformed = transformed.T
    stretch = stretch.T
    reconstructed = reconstructed.T # Transpose results to be in line with sklearn standard

    return X, transformed, reconstructed, errors, stretch

def CNMF_analysis(intensities, comp_num, *,
    beta,
    tol,
    max_iter,
    alpha,
    l1_ratio,
    calc_err,
    exp_win,
    W,
    W_fix,
    H,
    H_fix
):
    """
    Calls on the constrained NMF algorithm and performs a fit to the user's dataset
    If data contains negative values, lifts them
    """
    errors = None

    if len(H_fix) > comp_num:
        print(
            f"\tH_fix contains more values ({len(H_fix)}) "
            f"than components ({comp_num}) to analyze."
            f"Cropping to {comp_num} components"
        )
        H_fix = H_fix[:comp_num]
    if len(W_fix) > len(intensities):
        print(
            f"\tW_fix contains more values ({len(W_fix)}) "
            f"than scans ({len(intensities)}) to analyze. "
            f"Cropping to {len(intensities)} scans"
        )
        W_fix = W_fix[:len(intensities)]

    lift_factor = intensities.min()
    if lift_factor < 0:
        intensities -= lift_factor
        print("\tNegative value found in dataset. Lifting data above zero")

    if H is not None:
        if lift_factor < 0:
            H -= lift_factor
        H = [torch.tensor(component[None, :], dtype=torch.float) for component in H]
    if W is not None:
        W = [torch.tensor(score[None, :], dtype=torch.float) for score in W]
    """if calc_err:
        pass
    elif exp_win['enable']:
        pass
    else:"""
    cnmf_model = CNMF(
        intensities.shape,
        n_components=comp_num,
        initial_components=H,
        fix_components=H_fix,
        initial_weights=W,
        fix_weights=W_fix
    )
    loss = cnmf_model.fit(
        torch.tensor(
        intensities, dtype=torch.float),
        beta=beta,
        tol=tol,
        max_iter=max_iter,
        alpha=alpha,
        l1_ratio=l1_ratio
    )
    weights, components = cnmf_model.W.detach().numpy(), cnmf_model.H.detach().numpy()
    reconstructed = cnmf_model.reconstruct(components, weights)
    print(
        f"\tCNMF ({comp_num}) Beta divergence loss: "
        f"{loss[-1]:.6e} in {len(loss)} iterations"
    )

    if lift_factor < 0: # Lower data below zero again
        intensities += lift_factor
        components += lift_factor
        reconstructed += lift_factor

    return ComponentsResult(components), weights, reconstructed, errors, lift_factor

def MCRALS_analysis(intensities, comp_num, *,
    C,
    c_fix,
    ST,
    st_fix,
    #c_regr,
    #st_regr,
    #fit_kwargs,
    #c_fit_kwargs,
    #st_fit_kwargs,
    #c_constraints,
    #st_constraints,
    max_iter,
    #err_fcn,
    tol_increase,
    tol_n_increase,
    tol_err_change,
    tol_n_above_min,
    init_type,
    verbose=False
):
    """
    Calls on the MCR-ALS algorithm and performs a fit to the user's dataset
    """
    errors = None

    if verbose:
        logger = logging.getLogger('pymcr')
        logger.setLevel(logging.DEBUG)
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_format = logging.Formatter('%(message)s')
        stdout_handler.setFormatter(stdout_format)
        logger.addHandler(stdout_handler)
    if tol_increase<0:
        tol_increase=None
    if tol_n_increase<0:
        tol_n_increase=None
    if tol_n_above_min<0:
        tol_n_above_min=None

    lift_factor = intensities.min()
    if lift_factor < 0:
        intensities -= lift_factor
        print("\tNegative value found in dataset. Lifting data above zero")

    C_expand = np.random.rand(intensities.shape[0],comp_num) # (samples, components)
    ST_expand = np.random.rand(comp_num,intensities.shape[1]) # (components, features)

    # Expand scores
    if C is None:
        C = C_expand
    else:
        if C.shape[0] < intensities.shape[0]: # Expand samples in guessed scores
            print(
                f"\tFewer sample scores than {intensities.shape[0]} were provided. "
                "Expanding with random"
            )
            C_vcrop = C_expand[C.shape[0]:,:]
            C = np.vstack([C, C_vcrop])
        if C.shape[1] < comp_num: # Expand components in guessed scores
            print(f"\tFewer components than {comp_num} were provided. Expanding with random")
            C_hcrop = C_expand[:,C.shape[1]:]
            C = np.hstack([C, C_hcrop])
        elif C.shape[1] > comp_num:
            raise ValueError(
                f"Analysis to be done with {comp_num} components, "
                f"but scores provided {C.shape[1]}"
            )

    # Expand components
    if ST is None:
        ST = ST_expand
    else:
        if lift_factor < 0:
            ST -= lift_factor
        if ST.shape[0] < comp_num: # Expand components in guessed components
            print(f"\tFewer components than {comp_num} were provided. Expanding with random")
            ST_vcrop = ST_expand[ST.shape[0]:,:]
            ST = np.vstack([ST, ST_vcrop])
        elif ST.shape[0] > comp_num:
            raise ValueError(
                f"Analysis to be done with {comp_num} components, "
                f"but {ST.shape[0]} were provided"
            )

    # Set to None depending on given data
    match init_type:
        case "Components":
            C = None
            c_fix = None
        case "Scores":
            ST = None
            st_fix = None
        case "Both":
            if (st_fix is None) or (c_fix is None):
                raise ValueError(
                    "Both \"Fix Components\" and \"Fix Scores\" must be provided for \"Both\""
                )

    if c_fix is not None:
        if len(c_fix) < intensities.shape[0]: # Fill with up to samples
            c_fix.extend([0] * (intensities.shape[0]-len(c_fix)))

    if st_fix is not None:
        if len(st_fix) < comp_num: # Fill with zero up to comp_num
            st_fix.extend([0] * (comp_num-len(st_fix)))

    mcrals_model = McrAR(
        #c_regr,
        #st_regr,
        #fit_kwargs,
        #c_fit_kwargs,
        #st_fit_kwargs,
        #c_constraints,
        #st_constraints,
        max_iter=max_iter,
        #err_fcn,
        tol_increase=tol_increase,
        tol_n_increase=tol_n_increase,
        tol_err_change=tol_err_change,
        tol_n_above_min=tol_n_above_min
    )

    mcrals_model.fit(
         intensities,
         C=C,
         c_fix=c_fix,
         ST=ST,
         st_fix=st_fix,
         c_first=True,
         verbose=verbose,
         post_iter_fcn=None,
         post_half_fcn=None
    )

    transformed = mcrals_model.C_
    reconstructed = mcrals_model.D_
    print(
        f"\tMCR-ALS ({comp_num}) calculted error: "
        f"{mcrals_model.err[-1]:.6e} in {mcrals_model.n_iter} iterations"
    )

    if lift_factor < 0: # Lower data below zero again
        intensities += lift_factor
        mcrals_model.ST_ += lift_factor # components
        reconstructed += lift_factor

    return mcrals_model, transformed, reconstructed, errors, lift_factor
