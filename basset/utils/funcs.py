"""
Module with functions used in or across BaSSET 
"""

import numpy as np


def theta_to_q(angles, wavelength):
    """
    Converts a dataset from 2theta to Q

    Parameters
    ----------
    angles: float or ndarray
        Angle to be converted from 2theta to Q or
        2D array of shape (samples, features) with a dataset's angles
    wavelength: float
        Wavelength used for data collection, given in angstrom
    
    Returns
    -------
    ndarray
        2D array of shape (samples, features) with the input dataset converted to Q
    """
    return ((4*np.pi) / wavelength) * np.sin(np.deg2rad(angles)/2)

def normalize_dataset(intensities):
    """
    Normalizes a dataset

    Parameters
    ----------
    intensities: ndarray
        2D array of shape (samples, features) with a dataset's intensities
    
    Returns
    -------
    ndarray
        2D array of shape (samples, features) with the input dataset noralized
    """
    for i, intensity in enumerate(intensities):
        intensities[i] = intensity / intensity.max()
    return intensities
