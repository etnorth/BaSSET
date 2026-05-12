import numpy as np


def theta_to_Q(angles, wavelength):
    return ((4*np.pi) / wavelength) * np.sin(np.deg2rad(angles)/2)

def normalize_dataset(intensities):
    for i in range(len(intensities)):
        intensities[i] = intensities[i] / intensities[i].max()
    return intensities
