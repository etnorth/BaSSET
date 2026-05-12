import numpy as np
from sklearn.decomposition import PCA, NMF, FastICA
from diffpy.stretched_nmf.snmf_class import SNMFOptimizer


def PCA_analysis(intensities, numComponents, whiten, svd_solver, tol, iterated_power, n_oversamples, power_iteration_normalizer):
    n_components = min(10,min(np.shape(intensities))) # Setting this higher than the user's number ensures reporting of explained variances
    pca = PCA(n_components = n_components, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, n_oversamples=n_oversamples, power_iteration_normalizer=power_iteration_normalizer)
    X = pca.fit(intensities)
    transformed = pca.transform(intensities)
    reconstructed = pca.inverse_transform(transformed)

    return X, transformed, reconstructed

def NMF_analysis(intensities, numComponents, init, solver, beta_loss, tol, max_iter, alpha_W, alpha_H, l1_ratio, calc_err):
    lift_factor = intensities.min()
    if lift_factor < 0:
        intensities -= lift_factor
        print("Negative value found in dataset. Lifting data above zero")

    if calc_err:
        n_components_list = np.arange(1, min(10,min(np.shape(intensities)))+1, dtype=int)
        errors = np.empty(len(n_components_list))
        for i, n_components in enumerate(n_components_list):
            print(f"Calculating NMF reconstruction error for {n_components} components")
            nmf = NMF(n_components=n_components, init=init, solver=solver, beta_loss=beta_loss, tol=tol, max_iter=max_iter, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio)
            X_temp = nmf.fit(intensities)
            errors[i] = X_temp.reconstruction_err_
            print(f"    NMF ({n_components}) reconstruction error: {X_temp.reconstruction_err_:10f}")
            if numComponents == n_components:
                X = X_temp
                transformed = nmf.transform(intensities)
                reconstructed = nmf.inverse_transform(transformed)
    else:
        nmf = NMF(n_components=numComponents, init=init, solver=solver, beta_loss=beta_loss, tol=tol, max_iter=max_iter, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio)
        errors = None
        X = nmf.fit(intensities)
        transformed = nmf.transform(intensities)
        reconstructed = nmf.inverse_transform(transformed)
        print(f"    NMF ({numComponents}) reconstruction error: {X.reconstruction_err_:10f}")

    if lift_factor < 0: # Lower data below zero again
        intensities += lift_factor
        X.components_ += lift_factor
        reconstructed += lift_factor

    return X, transformed, reconstructed, errors, lift_factor

def ICA_analysis(intensities, numComponents, algorithm, whiten, fun, max_iter, tol, whiten_solver):
    ica = FastICA(n_components = numComponents, algorithm=algorithm, whiten=whiten, fun=fun, max_iter=max_iter, tol=tol, whiten_solver=whiten_solver)
    X = ica.fit(intensities)
    transformed = ica.transform(intensities)
    reconstructed = ica.inverse_transform(transformed)

    return X, transformed, reconstructed

def SNMF_analysis(intensities, numComponents, min_iter, max_iter, tol, rho, eta, calc_err):
    # SNMF automatically handles lifts negative values, so no if-case needed
    calc_err=False

    if calc_err:
        n_components_list = np.arange(1, min(10,min(np.shape(intensities)))+1, dtype=int)
        errors = np.empty(len(n_components_list))
        for i, n_components in enumerate(n_components_list):
            print(f"Calculating SNMF reconstruction error for {n_components} components")
            snmf = SNMFOptimizer(n_components=n_components, min_iter=min_iter, max_iter=max_iter, tol=tol, rho=rho, eta=eta, show_plots=True)
            X_temp = snmf.fit(intensities)
            errors[i] = X_temp.reconstruction_err_
            print(f"    SNMF ({n_components}) reconstruction error: {X_temp.reconstruction_err_:10f}")
            if numComponents == n_components:
                X = X_temp
                transformed = snmf.weights_
                stretch = snmf.stretch_
                reconstructed = snmf.reconstruct_matrix(snmf.components_, transformed, stretch)
    else:
        snmf = SNMFOptimizer(n_components=numComponents, min_iter=min_iter, max_iter=max_iter, tol=tol, rho=rho, eta=eta, show_plots=True, verbose=True)
        errors = None
        X = snmf.fit(intensities)
        transformed = snmf.weights_
        stretch = snmf.stretch_
        reconstructed = snmf.reconstruct_matrix(snmf.components_, transformed, stretch)
        print(f"    NMF ({numComponents}) reconstruction error: {X.reconstruction_err_:10f}")

    return X, transformed, reconstructed, errors, stretch
