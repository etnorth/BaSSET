"""
Module containing algorithm calls for BaSSET's data analysis
"""
# pylint: disable=invalid-name

import numpy as np
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.decomposition._nmf import _initialize_nmf
from diffpy.stretched_nmf.snmf_class import SNMFOptimizer, _reconstruct_matrix


def PCA_analysis(intensities, comp_num, *,
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
    pca_model = PCA(n_components=n_components,
              whiten=whiten,
              svd_solver=svd_solver,
              tol=tol,
              iterated_power=iterated_power,
              n_oversamples=n_oversamples,
              power_iteration_normalizer=power_iteration_normalizer)
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
                 rescale,
                 exp_win
):
    """
    Calls on sklearn's NMF algorithm and performs a fit to the user's dataset
    Calculates error for 1-10 components, and if data contains negative values, lifts them
    """
    lift_factor = intensities.min()
    if lift_factor < 0:
        intensities -= lift_factor
        print("Negative value found in dataset. Lifting data above zero")

    errors = None
    if calc_err:
        n_components_list = np.arange(1, min(10,*np.shape(intensities))+1, dtype=int)
        errors = np.empty(len(n_components_list))
        for i, n_components in enumerate(n_components_list):
            print(f"Calculating NMF reconstruction error for {n_components} components")
            nmf_model = NMF(n_components=n_components,
                      init=init,
                      solver=solver,
                      beta_loss=beta_loss,
                      tol=tol,
                      max_iter=max_iter,
                      alpha_W=alpha_W,
                      alpha_H=alpha_H,
                      l1_ratio=l1_ratio)
            nmf_model_temp = nmf_model.fit(intensities)
            errors[i] = nmf_model_temp.reconstruction_err_
            print(
                f"    NMF ({n_components}) reconstruction error: "
                f"{nmf_model_temp.reconstruction_err_:10f}"
            )
            if comp_num == n_components:
                nmf_model = nmf_model_temp
                transformed = nmf_model.transform(intensities)
                reconstructed = nmf_model.inverse_transform(transformed)
    elif exp_win['enable']:
        # Does not perform calc_err for 1-10 components
        if exp_win['do_custom']:
            if exp_win['win_custom'][-1] < len(intensities): # Makes sure it includes entire set
                exp_win['win_custom'] = np.hstack([exp_win['win_custom'],(len(intensities)+1)])
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
        nmf_model = NMF(n_components=win_comps[0],
                  init=init,
                  solver=solver,
                  beta_loss=beta_loss,
                  tol=tol,
                  max_iter=max_iter,
                  alpha_W=alpha_W,
                  alpha_H=alpha_H,
                  l1_ratio=l1_ratio)
        nmf_model = nmf_model.fit(intensities[:win_ends[0],:])
        W = nmf_model.transform(intensities[:win_ends[0],:])
        H = nmf_model.components_
        print(
            f"    NMF ({win_comps[0]}) [start to {win_ends[0]}] reconstruction error: "
            f"{nmf_model.reconstruction_err_:10f}"
        )

        for i, (win_end, win_comp) in enumerate(zip(win_ends[1:], win_comps[1:])):
            win_intensities = intensities[:win_end,:]
            W_rows_old, prev_comp = W.shape

            W_init, _ = _initialize_nmf( # Create new W-rows for larger window
                win_intensities,
                prev_comp,
                init=init
            )

            W_vcrop = W_init[W_rows_old:,:] # Get new rows for vstack
            W = np.vstack([W, W_vcrop]) # Add news row to old W
            if win_comp > prev_comp: # Fit residual of new window with old comps as new comps
                win_reconstructed = nmf_model.inverse_transform(W)

                resid = np.maximum(win_intensities - win_reconstructed, 0) # Non-negative residual

                # Fit residual as new comps
                nmf_model_resid = NMF(n_components=win_comp-prev_comp,
                  init=init,
                  solver=solver,
                  beta_loss=beta_loss,
                  tol=tol,
                  max_iter=max_iter,
                  alpha_W=alpha_W,
                  alpha_H=alpha_H,
                  l1_ratio=l1_ratio)
                nmf_model_resid = nmf_model_resid.fit(resid)
                W_resid = nmf_model_resid.transform(resid)
                H_resid = nmf_model_resid.components_

                W = np.hstack([W, W_resid])
                H = np.vstack([H, H_resid])
            # Perform NMF on current window with prev+resid components
            nmf_model = NMF(n_components=win_comp, # initialize with 'custom' for W and H guesses
                  init='custom',
                  solver=solver,
                  beta_loss=beta_loss,
                  tol=tol,
                  max_iter=max_iter,
                  alpha_W=alpha_W,
                  alpha_H=alpha_H,
                  l1_ratio=l1_ratio)
            nmf_model = nmf_model.fit(win_intensities, W=W, H=H)
            W = nmf_model.transform(win_intensities)
            H = nmf_model.components_
            print(
                f"    NMF ({win_comp}) [start to {win_end}] reconstruction error: "
                f"{nmf_model.reconstruction_err_:10f}"
            )

        transformed = W
        reconstructed = nmf_model.inverse_transform(transformed)

    else:
        nmf_model = NMF(n_components=comp_num,
                  init=init,
                  solver=solver,
                  beta_loss=beta_loss,
                  tol=tol,
                  max_iter=max_iter,
                  alpha_W=alpha_W,
                  alpha_H=alpha_H,
                  l1_ratio=l1_ratio)
        nmf_model = nmf_model.fit(intensities)
        transformed = nmf_model.transform(intensities)
        reconstructed = nmf_model.inverse_transform(transformed)
        print(f"    NMF ({comp_num}) reconstruction error: {nmf_model.reconstruction_err_:10f}")

    if lift_factor < 0: # Lower data below zero again
        intensities += lift_factor
        nmf_model.components_ += lift_factor
        reconstructed += lift_factor

    if rescale:
        print("Rescaling scores")
        row_sums = np.sum(transformed, axis=1, keepdims=True) # Sums scores
        transformed = transformed / row_sums * 1 # Rescales scores so they sum to one

    return nmf_model, transformed, reconstructed, errors, lift_factor

def ICA_analysis(intensities, comp_num, *,
                 algorithm,
                 whiten,
                 fun,
                 max_iter,
                 tol,
                 whiten_solver,
                 calc_err
):
    """
    Calls on sklearn's ICA algorithm and performs a fit to the user's dataset
    """
    ica_model = FastICA(n_components=comp_num,
                  algorithm=algorithm,
                  whiten=whiten,
                  fun=fun,
                  max_iter=max_iter,
                  tol=tol,
                  whiten_solver=whiten_solver)
    X = ica_model.fit(intensities)
    transformed = ica_model.transform(intensities)
    reconstructed = ica_model.inverse_transform(transformed)

    if calc_err:
        print("ICA does not return a numerical indication for goodness of fit")

    return X, transformed, reconstructed

def SNMF_analysis(intensities, comp_num, *,
                  min_iter,
                  max_iter,
                  tol,
                  rho,
                  eta,
                  calc_err
):
    """
    Calls on diffpy's stretched-nmf and performs a fit to the user's dataset
    """
    # SNMF automatically handles lifts negative values, so no if-case needed
    print("SNMF reconstruction error calculation is disabled due to slow convergence")

    intensities = intensities.T # SNMF: (n_features,n_samples), sklearn: (n_samples,n_features)

    if calc_err:
        n_components_list = np.arange(1, min(10,*np.shape(intensities))+1, dtype=int)
        errors = np.empty(len(n_components_list))
        for i, n_components in enumerate(n_components_list):
            print(f"Calculating SNMF reconstruction error for {n_components} components")
            snmf_model = SNMFOptimizer(n_components=n_components,
                                 min_iter=min_iter,
                                 max_iter=max_iter,
                                 tol=tol,
                                 rho=rho,
                                 eta=eta,
                                 show_plots=True)
            X_temp = snmf_model.fit(intensities)
            errors[i] = X_temp.reconstruction_err_
            print(
                f"    SNMF ({n_components}) reconstruction error: "
                f"{X_temp.reconstruction_err_:10f}"
            )
            if comp_num == n_components:
                X = X_temp
                transformed = snmf_model.weights_
                stretch = snmf_model.stretch_
                reconstructed = _reconstruct_matrix(snmf_model.components_, transformed, stretch)
    else:
        snmf_model = SNMFOptimizer(n_components=comp_num,
                             min_iter=min_iter,
                             max_iter=max_iter,
                             tol=tol,
                             rho=rho,
                             eta=eta,
                             show_plots=True,
                             verbose=True)
        errors = None
        X = snmf_model.fit(intensities)
        transformed = snmf_model.weights_
        stretch = snmf_model.stretch_
        reconstructed = _reconstruct_matrix(snmf_model.components_, transformed, stretch)
        print(f"    NMF ({comp_num}) reconstruction error: {X.reconstruction_err_:10f}")

    # Transpose to match sklearn
    X.components_ = X.components_.T
    transformed = transformed.T
    stretch = stretch.T
    reconstructed = reconstructed.T # Transpose results to be in line with sklearn standard

    return X, transformed, reconstructed, errors, stretch
