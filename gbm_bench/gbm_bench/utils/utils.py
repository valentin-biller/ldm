import os
# import ants
import shutil
import datetime
import platform
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from loguru import logger
# from PyPDF2 import PdfMerger
from contextlib import contextmanager
from scipy.ndimage import center_of_mass
from typing import List, Tuple, Optional, Union


def compute_center_of_mass(seg_data: np.ndarray, mri_data: np.ndarray, classes: List[int] = [1, 2, 3],) -> Tuple[int, int, int]:

    mask = np.isin(seg_data, classes)

    # Check if the mask contains any non-zero values (i.e., non-empty segmentation)
    if not np.any(mask):
        print("Warning: Segmentation is empty, returning middle slices of the MRI.")
        # Return the middle slices of the MRI volume as default
        return (mri_data.shape[0] // 2, mri_data.shape[1] // 2, mri_data.shape[2] // 2)

    # Compute center of mass if the segmentation is non-empty
    com = center_of_mass(mask)
    return tuple(map(int, com))


def load_mri_data(filepath: Union[Path, str]) -> np.ndarray:
    img = nib.load(str(filepath))
    data = img.get_fdata()
    return data


def load_and_resample_mri_data(filepath: Union[str, Path], resample_params: Tuple[int, int, int], interp_type: Optional[int] = 0,) -> np.ndarray:
    
    img = ants.image_read(str(filepath))
    img = ants.resample_image(
            image=img,
            resample_params=resample_params,
            use_voxels=True,
            interp_type=interp_type
            )
    return img.to_nibabel().get_fdata()


def make_symlink(src: Path, dst: Path) -> None:
    """
    Create a symlink `dst` → `src`, replacing an existing file if necessary.
    The symlink will point to the absolute path of `src`.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.unlink()  # Remove existing file or symlink if present
    except FileNotFoundError:
        pass

    # Resolve the absolute path of the source
    src_abs = src.resolve(strict=False)

    kwargs = {}
    if platform.system() == "Windows":
        kwargs["target_is_directory"] = src_abs.is_dir()
    dst.symlink_to(src_abs, **kwargs)


def merge_pdfs(pdf_list: List[Union[str, Path]], output_pdf: Union[str, Path]) -> None:
    pdf_merger = PdfMerger()

    for pdf in pdf_list:
        pdf_merger.append(str(pdf))

    pdf_merger.write(str(output_pdf))
    pdf_merger.close()
    print(f"Combined PDF saved as {str(output_pdf)}")


def is_binary_array(arr: np.ndarray) -> bool:
    allowed_values = {0, 1, 0.0, 1.0, False, True}
    return np.all(np.isin(arr, list(allowed_values)))


def remove_tmp_folder(folder: Union[str, Path]) -> None:
    """Remove a temporary folder and log a warning if it fails.

    Args:
        folder (Path): Path to the folder to be removed
    """
    try:
        shutil.rmtree(str(folder))
    except PermissionError as e:
        logger.warning(
            f"Failed to remove temporary folder {folder}. This is most likely caused by bad permission management of the docker container. \nError: {e}"
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to delete folder {folder}. {e}")


@contextmanager
def temporary_tmpdir(base_dir: Union[str, Path]) -> Path:
    """Create and clean up a temporary directory used as TMPDIR.

    All files written to the system temporary directory during the context
    lifetime will be redirected to this folder. In addition to setting the
    ``TMPDIR`` environment variable, this also updates ``tempfile.tempdir`` so
    libraries that cache the temporary directory respect the new location.
    The ``base_dir`` folder will be created if it does not already exist.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(dir=str(base_dir), prefix="tmp_"))
    logger.info(f"Created temporary directory at {tmpdir}")
    old_tmpdir = os.environ.get("TMPDIR")
    old_tempfile_dir = tempfile.tempdir
    os.environ["TMPDIR"] = str(tmpdir)
    tempfile.tempdir = str(tmpdir)
    try:
        yield tmpdir
    finally:
        if old_tmpdir is not None:
            os.environ["TMPDIR"] = old_tmpdir
        else:
            os.environ.pop("TMPDIR", None)
        tempfile.tempdir = old_tempfile_dir
        remove_tmp_folder(tmpdir)
        logger.info(f"Removed temporary directory at {tmpdir}")
