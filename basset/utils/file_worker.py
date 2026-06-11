"""
Module to handle importing and exporting of files
"""

import os
from glob import glob
from re import search

from natsort import natsorted
import numpy as np


def most_common_filetype(indir):
    """
    Finds the most common filetype in a given directory

    Parameters
    ----------
    indir: str
        Dirpath pointing to a directory of files

    Returns
    -------
    filetype: str
        Most common filetype in directory
    """
    filetypes = {}
    for file_extension in [os.path.splitext(filename)[-1] for filename in glob(f"{indir}*")]:
        # Gets extension of each filename in folder
        if file_extension in filetypes:
            filetypes[file_extension] += 1
        else:
            filetypes[file_extension] = 1
    filetype = max(filetypes)
    # Checks if there are multiple equally popular file extensions
    popular_filetypes = [key for key, value in filetypes.items() if value == filetypes[filetype]]
    if len(popular_filetypes)>1:
        raise NotImplementedError("Multiple equally popular filetypes found:" \
                                  f"({popular_filetypes}). Check dir and e-run")
    print(f"Most common filetype in supplied directory is {filetype}")
    return filetype

def get_dataset_details(indir: str):
    """
    Evaluates a subset of the chosen dataset in order to accurately set widget_limits

    Parameters
    ----------
    indir: str
       Dirpath pointing to a directory of two-column datasets

    Returns
    xmin: float
        The smallest x-value in the imported data
    xmax: float
        The largest x-value in the imported data
    num_scans: int
        The number of scans in the dataset
    """
    print("Getting dataset details from {indir}")
    if not indir.endswith(os.path.sep):
        indir += os.path.sep

    filetype = most_common_filetype(indir)
    filenames = natsorted(glob(f"{indir}*{filetype}"))

    x,_ = import_data(filenames[0])

    xmin = np.min(x)
    xmax = np.max(x)
    num_scans = len(filenames)
    print(
        f"Minimum angle: {xmin}\n"
        f"Maximum angle: {xmax}\n"
        f"Number of scans {num_scans}"
    )

    return xmin, xmax, num_scans

def import_data(filename):
    """
    Takes in a file to find continous two-column datasets and return them

    Parameters
    ----------
    filename: str
        The filepath pointing to the wanted two-column dataset
    
    Returns
    -------
    ndarray
        1D array containing angles in data
    ndarray
        1D array containing intensity for each given angle in data
    """
    try:
        x, y = np.loadtxt(filename, unpack=True, comments='#', usecols=(0,1))
        return x, y
    except ValueError:
        #print("Couldn't read file with default # comments, reading file line by line")
        pass
    with open(filename, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        for i, line in enumerate(lines): # Find first non-empty line beginning with a number
            if not line:
                continue
            if search("[0-9]", line.replace(' ','')[0]):
                # Checks if first non-whitespace symbol is a number and skips row if not
                x, y = np.loadtxt(filename, unpack=True, skiprows=i, usecols=(0,1))
                return x, y

        raise ValueError(f"Couldn't find data in {filename}")

def import_dataset(indir):
    """
    Imports data from all files in chosen directory of given filetype
    
    Parameters
    ----------
    indir: str
       Dirpath pointing to a directory of two-column datasets
    
    Returns
    -------
    ndarray
        2D array of shape (samples, features) containing angles in dataset
    ndarray
        2D array of shape (samples, features) containing intensity for each given angle in dataset
    """
    print("Importing dataset")
    if not indir.endswith(os.path.sep):
        indir += os.path.sep

    filetype = most_common_filetype(indir)

    print(f"Looking for files in \"...{indir[-41:-1]}\" of type {filetype}")
    filenames = natsorted(glob(f"{indir}*{filetype}"))
    print(f"Found {len(filenames)} files of filetype {filetype}")

    x_data = []
    y_data = []

    for filename in filenames:
        x, y = import_data(filename)
        x_data.append(x)
        y_data.append(y)

    print("Dataset imported\n")
    return np.array(x_data), np.array(y_data)

def import_scores(filename, comp_num):
    """
    Imports scores from BaSSET's generated scores.csv file
    
    Parameters
    ----------
    filename: str
        Filepath pointing to comma-separated n-column data file (scores.csv)
    comp_num: int
        Which component (column) to extract from scores file
    Returns
    -------
    ndarray
        1D array of chosen components score throughout it originating dataset
    """
    comp_scale = []
    with open(filename, 'r', encoding='utf-8') as infile:
        print(comp_num)
        for line in infile.readlines()[1:]:
            words = line.split(',')
            comp_scale.append(float(words[comp_num-1]))
    print("Scores imported")
    return np.array(comp_scale)

def write_components(angles, results_path, fitted):
    """
    Writes extracted components from analysis

    Parameters
    ----------
    angles: ndarray
        2D array of shape (samples, features) containing angles in dataset
    results_path: str
        Directory path to where results shall be saved
    fitted: ndarray
        2D array of shape (comp_num, features) containing intensity of extracted components            
    """
    os.mkdir(f"{results_path}/components")
    for i, component in enumerate(fitted.components_):
        with open(f"{results_path}/components/component_{i+1:02}.xy",
                  'w',
                  encoding='utf-8') as outfile:
            for j in range(len(angles[i])):
                outfile.write(f"{angles[i][j]}\t{component[j]}\n")
    print("Components written")

def write_scores(results_path, transformed):
    """
    Exports analysis component scores

    Parameters
    ----------
    results_path: str
        Directory path to where results shall be saved
    transformed: ndarray
        2D array of shape (samples, comp_num) containing scaling of extracted components
    """
    with open(f"{results_path}/scores.csv", 'w', encoding='utf-8') as outfile:
        for i in range(len(transformed[0])-1):
            outfile.write(f"Component {i+1:02},")
        outfile.write(f"Component {len(transformed[0]):02}\n")

        for i in range(len(transformed)):
            for j in range(len(transformed[0])-1):
                outfile.write(f"{transformed[i][j]},")
            outfile.write(f"{transformed[i][len(transformed[0])-1]}\n")
    print("Scores written")

def write_reconstructions(angles, results_path, reconstructed):
    """
    Exports analysis reconstruction of input dataset

    Parameters
    ----------
    angles: ndarray
        2D array of shape (samples, features) containing angles in dataset
    results_path: str
        Directory path to where results shall be saved
    reconstructed: ndarray
        2D array of shape (samples, features) containing reconstructed intensities from dataset
    """
    os.mkdir(f"{results_path}/reconstructions")
    for i in range(len(reconstructed)):
        with open(f"{results_path}/reconstructions/recon_scan_{i+1:04}.xy",
                  'w',
                  encoding='utf-8') as outfile:
            for j in range(len(angles[i])):
                outfile.write(f"{angles[i][j]}\t{reconstructed[i][j]}\n")
    print("Reconstructions written")

def write_differences(angles, intensities, results_path, reconstructed):
    """
    Exports difference between input data and resulting reconstruction

    Parameters
    ----------
    angles: ndarray
        2D array of shape (samples, features) containing angles in dataset
    intensities: ndarray
        2D array of shape (samples, features) containing intensities in datasett
    results_path: str
        Directory path to where results shall be saved
    reconstructed: ndarray
        2D array of shape (samples, features) containing reconstructed intensities from dataset
    """
    os.mkdir(f"{results_path}/differences")
    for i in range(len(reconstructed)):
        with open(f"{results_path}/differences/diff_scan_{i+1:04}.xy",
                  'w',
                  encoding='utf-8') as outfile:
            for j in range(len(angles[i])):
                outfile.write(f"{angles[i][j]}\t{intensities[i][j]-reconstructed[i][j]}\n")
    print("Differences written")

def write_comp_contribute(angles, results_path, fitted, transformed, stretch=None):
    """
    Creates and exports partial reconstruction through single component contributions

    Parameters
    ----------
    angles: ndarray
        2D array of shape (samples, features) containing angles in dataset
    results_path: str
        Directory path to where results shall be saved
    fitted: class
        sklearn/snmf class containing 'components_: ndarray'
        2D array of shape (comp_num, features) containing intensity of extracted components
    transformed: ndarray
        2D array of shape (samples, comp_num) containing scaling of extracted components
    stretch: ndarray
        2D array of shape (comp_num, features) containing angle stretch of extracted components
    """
    os.mkdir(f"{results_path}/component_contributions")

    if stretch is not None:
        for comp_num, component in enumerate(fitted.components_):
            os.mkdir(f"{results_path}/component_contributions/component_{comp_num+1:02}")
            for i in range(len(transformed)): # Number of scans
                with open(f"{results_path}/component_contributions/component_{comp_num+1:02}/c{comp_num+1:02}_contribution_scan_{i}.xy",
                          'w',
                          encoding='utf-8') as outfile:
                    for j in range(len(angles[comp_num])): # Index in scan
                        outfile.write(f"{angles[i][j]/stretch[i][comp_num]}\t{component[j]*transformed[i][comp_num]}\n") # Component multiplied by its scoring at that scan number
    else:
        for comp_num, component in enumerate(fitted.components_):
            os.mkdir(f"{results_path}/component_contributions/component_{comp_num+1:02}")
            for i in range(len(transformed)): # Number of scans
                with open(f"{results_path}/component_contributions/component_{comp_num+1:02}/c{comp_num+1:02}_contribution_scan_{i}.xy",
                          'w',
                          encoding='utf-8') as outfile:
                    for j in range(len(angles[comp_num])): # Index in scan
                        outfile.write(f"{angles[i][j]}\t{component[j]*transformed[i][comp_num]}\n")# Component multiplied by its scoring at that scan number
    print("Component contributions written")

def write_stretch(results_path, stretch):
    """
    Exports analysis component stretching

    Parameters
    ----------
    results_path: str
        Directory path to where results shall be saved
    stretch: ndarray
        2D array of shape (samples, comp_num) containing stretching of extracted components
    """
    with open(f"{results_path}/stretch.csv", 'w', encoding='utf-8') as outfile:
        for i in range(len(stretch[0])-1): # Due to comma separation, final comp is after loop
            outfile.write(f"Component {i+1:02},")
        outfile.write(f"Component {len(stretch[0]):02}\n")

        for i in range(len(stretch)):
            for j in range(len(stretch[0])-1): # Due to comma separation, final comp is after loop
                outfile.write(f"{stretch[i][j]},")
            outfile.write(f"{stretch[i][len(stretch[0])-1]}\n")
    print("Stretch written")
