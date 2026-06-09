"""
Module containing algorithm calls for BaSSET's data analysis
"""

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
                 power_iteration_normalizer):
    """
    Calls on sklearn's PCA algorithm and performs a fit to the user's dataset
    """
    n_components = min(10,*np.shape(intensities)) # Uses 10 for reporting explained variances
    pca = PCA(n_components=n_components,
              whiten=whiten,
              svd_solver=svd_solver,
              tol=tol,
              iterated_power=iterated_power,
              n_oversamples=n_oversamples,
              power_iteration_normalizer=power_iteration_normalizer)
    X = pca.fit(intensities)
    transformed = pca.transform(intensities)
    reconstructed = pca.inverse_transform(transformed)

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
                 exp_win):
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
            X = NMF(n_components=n_components,
                      init=init,
                      solver=solver,
                      beta_loss=beta_loss,
                      tol=tol,
                      max_iter=max_iter,
                      alpha_W=alpha_W,
                      alpha_H=alpha_H,
                      l1_ratio=l1_ratio)
            X_temp = X.fit(intensities)
            errors[i] = X_temp.reconstruction_err_
            print(f"    NMF ({n_components}) reconstruction error: {X_temp.reconstruction_err_:10f}")
            if comp_num == n_components:
                X = X_temp
                transformed = X.transform(intensities)
                reconstructed = X.inverse_transform(transformed)
    elif exp_win['enable']:
        # Does not perform calc_err for 1-10 components

        if exp_win['do_custom']:
            if exp_win['win_custom'][-1] < len(intensities): # Makes sure it includes entire set
                exp_win['win_custom'] = np.hstack([exp_win['win_custom'],(len(intensities)+1)])
            win_ends = exp_win['win_custom']
        else: # Creates num_win windows with uniform distribution
            win_ends = np.linspace(1,
                                   len(intensities),
                                   exp_win['num_win']+1,dtype=int
            )[1:] # Exclude start
        X = NMF(n_components=comp_num, # 'first' iteration outside loop, warm-start inside loop
                  init=init,
                  solver=solver,
                  beta_loss=beta_loss,
                  tol=tol,
                  max_iter=max_iter,
                  alpha_W=alpha_W,
                  alpha_H=alpha_H,
                  l1_ratio=l1_ratio)
        X = X.fit(intensities[:win_ends[0],:])
        W = X.transform(intensities[:win_ends[0],:])
        H = X.components_
        print(f"    NMF ({comp_num}) [start to {win_ends[0]}] reconstruction error: {X.reconstruction_err_:10f}")
        for i, win_end in enumerate(win_ends[1:]):
            win_intensities = intensities[:win_end,:]

            W_init, _ = _initialize_nmf( # Create new W-rows for larger window
                win_intensities,
                comp_num,
                init=init
            )
            W_init = W_init[len(W):,:] # Get only new rows
            W_init = np.vstack([W, W_init]) # Add news row to old W

            X = NMF(n_components=comp_num, # initialize with 'custom' for W and H guesses
                  init='custom',
                  solver=solver,
                  beta_loss=beta_loss,
                  tol=tol,
                  max_iter=max_iter,
                  alpha_W=alpha_W,
                  alpha_H=alpha_H,
                  l1_ratio=l1_ratio)
            X = X.fit(win_intensities, W=W_init, H=H)
            W = X.transform(win_intensities)
            H = X.components_
            print(f"    NMF ({comp_num}) [start to {win_end}] reconstruction error: {X.reconstruction_err_:10f}")
        transformed = W
        reconstructed = X.inverse_transform(transformed)
        print(f"    NMF ({comp_num}) reconstruction error: {X.reconstruction_err_:10f}")
    else:
        X = NMF(n_components=comp_num,
                  init=init,
                  solver=solver,
                  beta_loss=beta_loss,
                  tol=tol,
                  max_iter=max_iter,
                  alpha_W=alpha_W,
                  alpha_H=alpha_H,
                  l1_ratio=l1_ratio)
        X = X.fit(intensities)
        transformed = X.transform(intensities)
        reconstructed = X.inverse_transform(transformed)
        print(f"    NMF ({comp_num}) reconstruction error: {X.reconstruction_err_:10f}")

    if lift_factor < 0: # Lower data below zero again
        intensities += lift_factor
        X.components_ += lift_factor
        reconstructed += lift_factor

    if rescale:
        print("Rescaling scores")
        row_sums = np.sum(transformed, axis=1, keepdims=True) # Sums scores
        transformed = transformed / row_sums * 1 # Rescales scores so they sum to one

    return X, transformed, reconstructed, errors, lift_factor

def ICA_analysis(intensities, comp_num, *,
                 algorithm,
                 whiten,
                 fun,
                 max_iter,
                 tol,
                 whiten_solver,
                 calc_err):
    """
    Calls on sklearn's ICA algorithm and performs a fit to the user's dataset
    """
    ica = FastICA(n_components=comp_num,
                  algorithm=algorithm,
                  whiten=whiten,
                  fun=fun,
                  max_iter=max_iter,
                  tol=tol,
                  whiten_solver=whiten_solver)
    X = ica.fit(intensities)
    transformed = ica.transform(intensities)
    reconstructed = ica.inverse_transform(transformed)

    if calc_err:
        print(f"ICA does not return a numerical indication for goodness of fit")

    return X, transformed, reconstructed

def SNMF_analysis(intensities, comp_num, *,
                  min_iter,
                  max_iter,
                  tol,
                  rho,
                  eta,
                  calc_err):
    # SNMF automatically handles lifts negative values, so no if-case needed
    print("SNMF reconstruction error calculation is disabled due to slow convergence")

    intensities = intensities.T # SNMF: (n_features,n_samples), sklearn: (n_samples,n_features)

    if calc_err:
        n_components_list = np.arange(1, min(10,*np.shape(intensities))+1, dtype=int)
        errors = np.empty(len(n_components_list))
        for i, n_components in enumerate(n_components_list):
            print(f"Calculating SNMF reconstruction error for {n_components} components")
            snmf = SNMFOptimizer(n_components=n_components,
                                 min_iter=min_iter,
                                 max_iter=max_iter,
                                 tol=tol,
                                 rho=rho,
                                 eta=eta,
                                 show_plots=True)
            X_temp = snmf.fit(intensities)
            errors[i] = X_temp.reconstruction_err_
            print(f"    SNMF ({n_components}) reconstruction error: {X_temp.reconstruction_err_:10f}")
            if comp_num == n_components:
                X = X_temp
                transformed = snmf.weights_
                stretch = snmf.stretch_
                reconstructed = _reconstruct_matrix(snmf.components_, transformed, stretch)
    else:
        snmf = SNMFOptimizer(n_components=comp_num,
                             min_iter=min_iter,
                             max_iter=max_iter,
                             tol=tol,
                             rho=rho,
                             eta=eta,
                             show_plots=True,
                             verbose=True)
        errors = None
        X = snmf.fit(intensities)
        transformed = snmf.weights_
        stretch = snmf.stretch_
        reconstructed = _reconstruct_matrix(snmf.components_, transformed, stretch)
        print(f"    NMF ({comp_num}) reconstruction error: {X.reconstruction_err_:10f}")

    # Transpose to match sklearn
    X.components_ = X.components_.T
    transformed = transformed.T
    stretch = stretch.T
    reconstructed = reconstructed.T # Transpose results to be in line with sklearn standard

    return X, transformed, reconstructed, errors, stretch
