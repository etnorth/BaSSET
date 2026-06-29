"""
Module to handle importing and exporting of files
"""
# pylint: disable=consider-using-enumerate

import os
from glob import glob
from re import search

from natsort import natsorted
import numpy as np


def most_common_filetype(indir: str, verbose=True):
    """
    Finds the most common filetype in a given directory

    Parameters
    ----------
    indir: str
        Dirpath pointing to a directory of files
    verbose: bool
        Controls print statements

    Returns
    -------
    filetype: str
        Most common filetype in directory
    """
    if len(glob(f"{indir}*")) == 0:
        print(f"\tFound {len(glob(f"{indir}*"))} files in {indir}")
        return None
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
    popular_filetypes = [x for x in popular_filetypes if x] # Removes empty strings (aka. folders)
    if len(popular_filetypes)>1:
        print(
            f"Multiple equally popular filetypes found: ({popular_filetypes}).\n"
            f"Assuming {filetype}. If unwanted, check dir and re-run"
        )

    if verbose:
        print(f"\tMost common filetype: {filetype}")

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
    if not indir.endswith(os.path.sep):
        indir += os.path.sep

    filetype = most_common_filetype(indir, verbose=False)
    filenames = natsorted(glob(f"{indir}*{filetype}"))

    x,_ = import_data(filenames[0])

    xmin = np.min(x)
    xmax = np.max(x)
    num_scans = len(filenames)

    return xmin, xmax, num_scans

def import_data(filename: str):
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

def import_dataset(indir: str):
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

    print(f"\tLooking for files in \"...{indir[-41:-1]}\" of type {filetype}")
    filenames = natsorted(glob(f"{indir}*{filetype}"))
    print(f"\tFound {len(filenames)} files of filetype {filetype}")

    x_data = []
    y_data = []

    for filename in filenames:
        x, y = import_data(filename)
        x_data.append(x)
        y_data.append(y)

    print("Dataset imported")
    return np.array(x_data), np.array(y_data)

def import_score(filename: str, comp_num: int):
    """
    Imports a components' score from BaSSET's generated scores.csv file
    
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
        for line in infile.readlines()[1:]:
            words = line.split(',')
            comp_scale.append(float(words[comp_num-1]))
    print("Scores imported")
    return np.array(comp_scale)

def import_init_guess(indir: str):
    """
    Imports provided initial guess of components and scores

    Parameters
    ----------
    indir: str
        Directory containing files for initial guess (components and/or scores)

    Returns
    -------
    init_components: ndarray
        2D array of shape (<=comp_num, features)
    init_scores: ndarray
        2D array of shape (<=samples, <=comp_num)

    """
    print("Importing initial guess")
    if not indir.endswith(os.path.sep):
        indir += os.path.sep

    filetype = most_common_filetype(indir)

    print(f"\tLooking for component files in \"...{indir[-41:-1]}\" of type {filetype}")
    filenames = natsorted(glob(f"{indir}*{filetype}"))
    print(f"\tFound {len(filenames)} files of filetype {filetype}")

    if len(filenames)==0:
        print("No components found, will initialize according to algorithm parameters")
        init_components = None
    elif len(filenames)==1:
        print("Assuming singular file is scores, not component")
        init_components = None
    else:
        init_components = []
        for filename in filenames:
            _, y = import_data(filename)
            init_components.append(y)
        init_components = np.array(init_components)
        print(f"Imported {len(init_components)} components")

    print("Looking for scores file")
    scoresfile = glob(f"{indir}*.csv")

    if len(scoresfile) > 1:
        raise ValueError(f"Expected 1 file of type csv, but found {len(scoresfile)}")

    if len(scoresfile)==0:
        print("No scores found, will initialize according to algorithm parameters")
        init_scores = None
    else:
        init_scores = np.loadtxt(scoresfile[0], skiprows=1, delimiter=',')
        print("Scores imported")

    return init_components, init_scores

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
        2D array of shape (samples, comp_num) containing stretching of extracted components
    """
    os.mkdir(f"{results_path}/component_contributions")

    if stretch is not None:
        for comp_num, component in enumerate(fitted.components_):
            os.mkdir(f"{results_path}/component_contributions/component_{comp_num+1:02}")
            for i in range(len(transformed)): # Number of scans
                # Recasts scaled component from stretched to original grid
                component_adjusted = (
                    np.interp(
                        angles[i] / stretch[i][comp_num], # current grid
                        angles[i], # new grid
                        component, # component (y-values)
                        left=component[0],
                        right=component[-1],
                    )
                    * transformed[i][comp_num] # scale component
                )
                with open(f"{results_path}/component_contributions/"
                          f"component_{comp_num+1:02}/c{comp_num+1:02}_contribution_scan_{i}.xy",
                          'w',
                          encoding='utf-8') as outfile:
                    for j in range(len(angles[i])): # Index in scan
                        outfile.write(
                            f"{angles[i][j]}"
                            f"\t{component_adjusted[j]}\n"
                        )
    else:
        for comp_num, component in enumerate(fitted.components_):
            os.mkdir(f"{results_path}/component_contributions/component_{comp_num+1:02}")
            for i in range(len(transformed)): # Number of scans
                with open(f"{results_path}/component_contributions/component_"
                          f"{comp_num+1:02}/c{comp_num+1:02}_contribution_scan_{i}.xy",
                          'w',
                          encoding='utf-8') as outfile:
                    for j in range(len(angles[i])): # Index in scan
                        outfile.write(
                            f"{angles[i][j]}"
                            f"\t{component[j]*transformed[i][comp_num]}\n"
                        )# Component multiplied by its scoring at that scan number
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
