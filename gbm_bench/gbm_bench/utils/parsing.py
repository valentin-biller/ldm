import json
import argparse
from pathlib import Path
from loguru import logger
from typing import List, Literal, Optional, TypedDict, Union
from gbm_bench.utils.constants import RHUH_GBM_DIR


class Exam(TypedDict):
    """
    Data type based on TypedDict that has modality names as keys and paths as values.
    """
    t1: Path
    t1c: Path
    t2: Path
    flair: Path
    pet: Optional[Path]
    diffusion: Optional[Path] = None
    perfusion: Optional[Path] = None
    tumorseg: Optional[Path] = None
    timepoint: Optional[Literal["preop", "postop", "followup"]]


class Patient(TypedDict):
    """
    Data type based on TypedDict that holds a patient id and its Exam objects.
    """
    patient_id: str
    patient_dir: Path
    exams: List[Exam]


class LongitudinalDataset():
    """
    Class for organizing paths from a longitudinal MRI dataset. Holds patients which in turn hold exams and
    provides saving/loading functionality via json and parsing for datasets with a specific diretory structure.
    """
    def __init__(self, dataset_id: str, root_dir: Union[str, Path]):
        root_dir = Path(root_dir).resolve()
        if not root_dir.is_dir():
            raise NotADirectoryError(f"Provided root_dir {root_dir} is not a valid directory.")

        self.dataset_id = dataset_id
        self.root_dir = root_dir
        self.patients = []

    def _convert_path(self, path: Optional[Path]) -> Optional[str]:
        """
        Helper function that converts Path objects to str while leaving None values as is.
        """
        return str(path) if path is not None else None

    def _substitute_root(self, substitute_path: Path, old_root: Path, new_root: Path) -> Path:
        """
        Helper function that replaces part of a Path object with a specified subpath.
        """
        substitu_str = str(substitute_path)
        old_root_str = str(old_root).strip("/")
        new_root_str = str(new_root).strip("/")
        return Path(substitu_str.replace(old_root_str, new_root_str))

    def parse(self):
        """
        Parses starting at root_dir assuming a directory structure root_dir/patient/exam/modality.ext
        Validates that the required modality files exist for each exam.
        """
        self.patients = []
        valid_exams = 0

        # Loop patients
        for patient_dir in self.root_dir.iterdir():
            if not patient_dir.is_dir():
                continue

            # Loop exams
            exams: List[Exam] = []
            for exam_dir in patient_dir.iterdir():
                if not exam_dir.is_dir():
                    continue

                # Check required modalities are present
                required_modalities = {
                    "t1": exam_dir / "t1.nii.gz",
                    "t1c": exam_dir / "t1c.nii.gz",
                    "t2": exam_dir / "t2.nii.gz",
                    "flair": exam_dir / "flair.nii.gz",
                }
                if any(not path.exists() for path in required_modalities.values()):
                    continue

                # Add paths to patients exam if they exist
                optional_modalities = {
                    "pet": exam_dir / "pet.nii.gz",
                    "diffusion": exam_dir / "diffusion.nii.gz",
                    "perfusion": exam_dir / "perfusion.nii.gz",
                    "tumorseg": exam_dir / "tumorseg.nii.gz"
                }
                exam_data = {
                        "t1": required_modalities["t1"],
                        "t1c": required_modalities["t1c"],
                        "t2": required_modalities["t2"],
                        "flair": required_modalities["flair"],
                        "pet": path if (path := optional_modalities["pet"]).exists() else None,
                        "diffusion": path if (path := optional_modalities["diffusion"]).exists() else None,
                        "perfusion": path if (path := optional_modalities["perfusion"]).exists() else None,
                        "tumorseg": path if (path := optional_modalities["tumorseg"]).exists() else None
                        }
                exams.append(exam_data)
                valid_exams += 1

            patient = {
                    "patient_id": patient_dir.name,
                    "patient_dir": patient_dir.resolve(),
                    "exams": exams,
                    }
            self.patients.append(patient)

        if valid_exams > 0:
            logger.info(f"Finished parsing {self.root_dir}. Found {valid_exams} valid exams.")
        else:
            logger.warning(f"Finished parsing but found {valid_exams} valid exams. Make sure the directory structure is correct and files are t1.nii.gz, ...")


    def save(self, out: Union[str, Path]) -> None:
        """
        Saves dataset by converting it to a dict and saving as json. Path objects are converted to strings
        for readability.
        """
        out = Path(out)
        if out.suffix != ".json":
            raise ValueError(f"Invalid out file specified {out}. Should be a .json file.")

        # Convert data to a single dict
        data_dict = {
                "dataset_id": self.dataset_id,
                "root_dir": str(self.root_dir),
                "patients": [
                    {
                        "patient_id": p["patient_id"],
                        "patient_dir": str(p["patient_dir"]),
                        "exams": [
                            {
                                "t1": self._convert_path(e["t1"]),
                                "t1c": self._convert_path(e["t1c"]),
                                "t2": self._convert_path(e["t2"]),
                                "flair": self._convert_path(e["flair"]),
                                "pet": self._convert_path(e["pet"]),
                                "diffusion": self._convert_path(e["diffusion"]),
                                "perfusion": self._convert_path(e["perfusion"]),
                                "timepoint": e["timepoint"]
                                }
                            for e in p["exams"]
                            ]
                        }
                    for p in self.patients
                    ]
                }

        # Write json
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(data_dict, f, indent=2)

        logger.info(f"Dataset saved successfully to {str(out)}.")

    def load(self, path: Union[str, Path]):
        """
        Loads dataset from a json object as created by the save method.
        """
        path = Path(path)
        if not path.is_file() or path.suffix != ".json":
            raise ValueError(f"Provided path {str(path)} is not a valid json file.")

        self.patients = []
        old_root = self.root_dir

        # Read json
        with open(path, "r") as f:
            data = json.load(f)

        # Fill attributes
        self.dataset_id = data["dataset_id"]
        self.root_dir = Path(data["root_dir"])
        self.patients = []

        for p_data in data["patients"]:
            exams = []
            for e_data in p_data["exams"]:
                exam  = {
                        "t1": Path(e_data["t1"]),
                        "t1c": Path(e_data["t1c"]),
                        "t2": Path(e_data["t2"]),
                        "flair": Path(e_data["flair"]),
                        "pet": Path(e_data["pet"]) if ("pet" in e_data.keys() and e_data["pet"] is not None) else None,
                        "diffusion": Path(e_data["diffusion"]) if ("diffusion" in e_data.keys() and e_data["diffusion"] is not None) else None,
                        "perfusion": Path(e_data["perfusion"]) if ("perfusion" in e_data.keys() and e_data["perfusion"] is not None) else None,
                        "tumorseg": Path(e_data["tumorseg"]) if ("tumorseg" in e_data.keys() and e_data["tumorseg"] is not None) else None,
                        "timepoint": e_data["timepoint"]
                        }
                exams.append(exam)

            patient: Patient = {
                "patient_id": p_data["patient_id"],
                "patient_dir": Path(p_data["patient_dir"]),
                "exams": exams,
            }
            self.patients.append(patient)

        # Set old root dir and adapt paths if necessary
        if old_root != self.root_dir:
            self.set_root_dir(new_root_dir=old_root)

        logger.info(f"Successfully loaded {len(self.patients)} patients from {str(path)}.")

    def set_root_dir(self, new_root_dir: Union[Path, str]) -> None:
        """
        Updates the root directory and adjusts all patient and exam paths by string substituation.
        """
        # Update self.root_dir
        new_root = Path(new_root_dir).resolve()
        old_root = self.root_dir.resolve()
        self.root_dir = new_root

        # Loop patients
        for patient in self.patients:
            
            # Update patient_dir
            patient['patient_dir'] = self._substitute_root(patient['patient_dir'], old_root, new_root)
            
            # Loop exams
            for exam in patient['exams']:
                for modality, path in exam.items():
                    if isinstance(path, Path):
                        exam[modality] = self._substitute_root(exam[modality], old_root, new_root)
        
        logger.info(f"New root_dir set successfully.")

    def get_patient_exams(self, patient_id: str, timepoint: Optional[str] = None) -> List[Exam]:
        """
        Retrieves exams for a specific patient, optionally filtered by timepoint (None, preop, postop, followup).
        """
        for patient in self.patients:
            if patient["patient_id"] == patient_id:
                if timepoint is None:
                    return patient["exams"].copy()

                filtered_exams = []
                for exam in patient["exams"]:
                    # Use t1 path to derive exam directory name
                    if exam["timepoint"] == timepoint:
                        filtered_exams.append(exam)
                return filtered_exams
        return []


if __name__=="__main__":
    # Example
    # python gbm_bench/utils/parsing.py -rhuh_root /home/home/lucas/data/RHUH-GBM/Images/DICOM/RHUH-GBM
    parser = argparse.ArgumentParser()
    parser.add_argument("-rhuh_root", type=str, help="Path to the directory in the RHUH dataset containing the patient folders.")
    args = parser.parse_args()

    # Read RHUH
    rhuh_gbm = LongitudinalDataset(dataset_id="RHUH", root_dir="")
    rhuh_gbm.load(RHUH_GBM_DIR)

    # Save to temporary folder
    out_tmp = "tmp_parsing/rhuh.json"
    rhuh_gbm.save(out_tmp)

    # Load and compare
    rhuh_gbm1 = LongitudinalDataset(dataset_id="RHUH", root_dir="")
    rhuh_gbm1.load(out_tmp)
    is_restored = (rhuh_gbm.__dict__ == rhuh_gbm1.__dict__)
    if is_restored:
        print(f"Saved and restored dataset successfully to {out_tmp}.")
    else:
        raise ValueError(f"Dataset restoration failed. Loaded dataset is not the same as before saving.")

    # Try replacing root dir
    out_tmp_newroot = "tmp_parsing/rhuh_newroot.json"
    rhuh_gbm.set_root_dir("/test/rootdir/")
    rhuh_gbm.save(out_tmp_newroot)
