import os
from glob import glob
from re import search

from natsort import natsorted
import numpy as np


def import_data(filename):
    """
    Takes in a filename (str) to find continous two-column datasets, returning each column as a numpy array
    """
    try:
        x, y = np.loadtxt(filename, unpack=True, comments='#', usecols=(0,1))
        return x, y
    except ValueError:
        #print("Couldn't read file with default # comments, reading file line by line")
        pass
    with open(filename, "r") as infile:
        lines = infile.readlines()
        for i, line in enumerate(lines):
            if not line:
                continue
            if search("[0-9]", line.replace(' ','')[0]): # Checks if the first non-whitespace symbol is a number
                x, y = np.loadtxt(filename, unpack=True, skiprows=i, usecols=(0,1)) # Skips all lines above curren
                return x, y
        raise ValueError(f"Couldn't find data in {filename}")

def import_dataset(dir):
    """
    Imports data from all files in chosen directory of chosen / most popular filetype
    """
    if not dir.endswith(os.path.sep):
        dir += os.path.sep

    filetypes = {}
    for file_extension in [os.path.splitext(filename)[-1] for filename in glob(f"{dir}*")]: # Gets extension of each filename in folder
        if file_extension in filetypes:
            filetypes[file_extension] += 1
        else:
            filetypes[file_extension] = 1
    filetype = max(filetypes)
    popular_filetypes = [key for key, value in filetypes.items() if value == filetypes[filetype]] # Checks if there are multiple equally popular file extensions
    if len(popular_filetypes)>1:
        print(f"Multiple equally popular filetypes found: ({popular_filetypes}). Disable 'auto' and re-run")
        raise NotImplementedError
    print(f"Most common filetype in supplied directory is {filetype}")

    print(f"Looking for files in \"...{dir[-41:-1]}\" of type {filetype}")
    filenames = natsorted(glob(f"{dir}*{filetype}"))
    print(f"Found {len(filenames)} files of filetype {filetype}")

    x_data = []
    y_data = []

    for filename in filenames:
        x, y = import_data(filename)
        x_data.append(x)
        y_data.append(y)

    print("Dataset imported\n")
    return np.array(x_data), np.array(y_data)

def import_scores(filename, compNum):
    comp_scale = []
    with open(filename, 'r') as infile:
        for line in infile.readlines()[1:]:
            words = line.split(',')
            comp_scale.append(float(words[compNum-1]))
    print("Scores imported")
    return np.array(comp_scale)
