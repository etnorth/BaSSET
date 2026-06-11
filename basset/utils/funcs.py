"""
Module with functions used in or across BaSSET 
"""

import numpy as np


def theta_to_q(angle, wavelength):
    """
    Converts a dataset from 2theta to Q

    Parameters
    ----------
    angle: float or ndarray
        Angle to be converted from 2theta to Q or
        array with dataset angles
    wavelength: float
        Wavelength used for data collection, given in angstrom
    
    Returns
    -------
    ndarray
        Array with input dataset converted to Q
    """
    return ((4*np.pi) / wavelength) * np.sin(np.deg2rad(angle)/2)

def normalize_dataset(dataset):
    """
    Normalizes a dataset

    Parameters
    ----------
    dataset: ndarray
        Array of dataset values
    
    Returns
    -------
    ndarray
        Array of dataset normalized to largest dataset value
    """
    return dataset / np.max(dataset)
