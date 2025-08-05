import os
import shlex
import argparse
import subprocess
from typing import List
from pathlib import Path
from loguru import logger
from auxiliary.turbopath.turbopath import turbopath
from gbm_bench.utils.constants import DCM2NIIX_LOCATION


def remove_postfixes(export_dir: Path) -> None:
    """
    Remove postfixes created by dcm2niix (e.g. '_real') from filenames in the given directory.

    Parameters:
        outdir (Path): The directory containing files to be renamed.
    """
    # Iterate over all files in the directory.
    for f in export_dir.iterdir():
        if f.is_file() and ("_" in f.name) and not f.name.endswith(".log"):
            # Split the filename into the base (with possible postfixes) and extension(s)
            parts = f.name.split(".")
            # Extract the modality (before the first underscore)
            modality = parts[0].split("_")[0]
            # Construct the new filename using the modality and the original extensions
            new_name = ".".join([modality] + parts[1:])
            new_path = export_dir / new_name
            while os.path.isfile(new_path):  # prevent overiding, in case a dicom has to be converted manually
                new_path = new_path.split(".")[0] + "_a" + ".nii.gz"
            f.rename(new_path)
            logger.info(f"Renamed postfix file {f} to {new_path}.")


def convert_nifti(input_dir: Path, outfile: Path, dcm2niix_location: Path = DCM2NIIX_LOCATION) -> None:
    """
    Convert DICOM files to NIfTI format using dcm2niix.

    Parameters:
        input_dir (Path): Directory containing the raw DICOM files.
        outfile (Path): The full path for the output file.
        dcm2niix_location (Path, optional): The location of the dcm2niix executable.
    """
    try:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        cmd_readable = (
            str(dcm2niix_location)
            + " -d 9 -f "
            + str(outfile.name)
            + " -z y -o"
            + ' "'
            + str(outfile.parent)
            + '" "'
            + str(input_dir)
            + '"'
        )

        logger.info(f"Running: {cmd_readable}")
        cmd = shlex.split(cmd_readable)

        log_file = outfile.parent / f"{outfile.name}_conversion.log"
        with open(log_file, "w") as logf:
            subprocess.run(cmd, stdout=logf, stderr=logf)

        remove_postfixes(outfile.parent)
        logger.debug(f"Nifti conversion complete for {input_dir}.")

    except Exception as e:
        logger.error(f"Error while trying to convert {input_dir} via {cmd_readable}: {e}")


if __name__ == "__main__":
    # Example
    # python gbm_bench/preprocessing/dicom_to_nifti.py -dcm2niix_loc /home/home/lucas/bin/dcm2niix
    parser = argparse.ArgumentParser()
    parser.add_argument("-dcm2niix_loc", type=str, help="Path to your dcm2niix executable.")
    args = parser.parse_args()

    convert_nifti(
        input_dir=Path("test_data/exam1/t1c"),
        outfile=Path("./tmp_test_dcm2nii/t1c"),
        dcm2niix_location=args.dcm2niix_loc
    )
