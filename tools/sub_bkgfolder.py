import os
from glob import glob
from re import search

from basset.utils.fileWorker import (
    import_data,
    import_dataset,
    import_scores
)

from natsort import natsorted
import numpy as np

DATASET_DIR        = r"C:\Users\izarc\20240316_1445_time_sliced"
SCORES_FILEPATH    = r"C:\Users\erlendtn\OneDrive - Universitetet i Oslo\Eira PhD\Articles\4 BaSSET\data\NaBi battery\XRD\generated_realistic\output_files\BaSSET_results\260511-151551_NMF_4\scores.csv"
COMPONENT_FILEPATH = r"C:\Users\erlendtn\OneDrive - Universitetet i Oslo\Eira PhD\Articles\4 BaSSET\data\NaBi battery\XRD\generated_realistic\output_files\BaSSET_results\260511-151551_NMF_4\components\component_01.xy"
COMPONENT_DIR      = r"C:\Users\izarc\20240316_1445_time_sliced\BaSSET_results\260511-170130_NMF_2\reconstructions"

FILE_OR_DIR        = 'dir' # 'file' or 'dir'

COMPNUM            = 'auto' # [int] component number or 'auto' to get from component filename
OUTDIR             = 'auto' # [str] A valid directory path or 'auto' to use same folder as scores

EXPORT_SCALEDCOMP  = True   # [bool]


def scaleComp(comp_y, comp_scale):
    scaledComp = np.empty((len(comp_scale),len(comp_y)))
    for i, scale in enumerate(comp_scale):
        for j in range(len(comp_y)):
                scaledComp[i][j] = comp_y[j]*scale
        if i==0:
            print(f"Y scaled by {scale}, giving {scaledComp[i][-1]} at last index")
    print("Component scaled")
    return scaledComp

def write_scaledComp(comp_x, scaledComp, compNum, outDir):
    scaledCompDir = f"{outDir}{os.path.sep}scaledComp_{compNum}"
    os.mkdir(scaledCompDir)
    for i in range(len(scaledComp)):
        with open(f"{scaledCompDir}{os.path.sep}scaledComp_scan_{i:02}.xy", 'w') as outfile:
            for j in range(len(comp_x)):
                outfile.write(f"{comp_x[j]}\t{scaledComp[i][j]}\n")
    print("Scaled component exported")
    return None

def write_bkgFileSubtracted(dataset_x, dataset_y, scaledComp, compNum, outDir):
    bkgSubDir = f"{outDir}{os.path.sep}bkgSubComp_{compNum}"
    os.mkdir(bkgSubDir)
    for i in range(len(dataset_x)):
        with open(f"{bkgSubDir}{os.path.sep}bkgSub_scan_{i:02}.xy", 'w') as outfile:
            for j in range(len(dataset_x[i])):
                outfile.write(f"{dataset_x[i][j]}\t{dataset_y[i][j]-scaledComp[i][j]}\n")
    print("Scaled component subtracted dataset exported")
    return None

def write_bkgDirSubtracted(dataset_x, dataset_y, comp_y, outDir):
    bkgSubDir = f"{outDir}{os.path.sep}bkgSubDir"
    os.mkdir(bkgSubDir)
    for i in range(len(dataset_x)):
        with open(f"{bkgSubDir}{os.path.sep}bkgSub_scan_{i:02}.xy", 'w') as outfile:
            for j in range(len(dataset_x[i])):
                outfile.write(f"{dataset_x[i][j]}\t{dataset_y[i][j]-comp_y[i][j]}\n")
    print("Directory subtracted dataset exported")
    return None


def main():
    global COMPNUM
    global OUTDIR
    if COMPNUM=='auto':
        COMPNUM = int("".join([char for char in os.path.splitext(os.path.basename(COMPONENT_FILEPATH))[0] if char.isdigit()]))

    """
    Due to slow operation for large datasets, should be re-written
    to work per-scan rather than loading everything into memory first
    """

    dataset_x, dataset_y = import_dataset(DATASET_DIR)
    if FILE_OR_DIR=='file':
        if OUTDIR=='auto':
            OUTDIR = os.path.dirname(SCORES_FILEPATH)
        comp_x, comp_y = import_data(COMPONENT_FILEPATH)
        print("Component imported")

        if len(comp_x) < len(dataset_x[0]):
            print("Component is a cropped section of the dataset\n" \
                "Cropping dataset to match"
            )
            xmin_index = np.searchsorted(dataset_x[0], comp_x[0], side='left')
            xmax_index = np.searchsorted(dataset_x[0], comp_x[-1], side='right')
            dataset_x = dataset_x[:,xmin_index:xmax_index]
            dataset_y = dataset_y[:,xmin_index:xmax_index]

        comp_scale = import_scores(SCORES_FILEPATH, COMPNUM)
        scaledComp = scaleComp(comp_y, comp_scale)
        write_scaledComp(comp_x, scaledComp, COMPNUM, OUTDIR) if EXPORT_SCALEDCOMP else None
        write_bkgFileSubtracted(dataset_x, dataset_y, scaledComp, COMPNUM, OUTDIR)
    elif FILE_OR_DIR=='dir':
        if OUTDIR=='auto':
            OUTDIR = os.path.dirname(COMPONENT_DIR)
        comp_x, comp_y = import_dataset(COMPONENT_DIR)

        if len(comp_x[0]) < len(dataset_x[0]):
            print("Component is a cropped section of the dataset\n" \
                "Cropping dataset to match"
            )
            xmin_index = np.searchsorted(dataset_x[0], comp_x[0][0], side='left')
            xmax_index = np.searchsorted(dataset_x[0], comp_x[0][-1], side='right')
            dataset_x = dataset_x[:,xmin_index:xmax_index]
            dataset_y = dataset_y[:,xmin_index:xmax_index]

        write_bkgDirSubtracted(dataset_x, dataset_y, comp_y, OUTDIR)

if __name__=="__main__":
    main()