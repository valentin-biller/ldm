import os
import argparse
from pathlib import Path
from loguru import logger
from typing import Optional
from gbm_bench.prediction.growth_model import TumorGrowthModel
from gbm_bench.utils.constants import MODALITY_STRIPPED_SCHEMA, TISSUE_SCHEMA, TISSUE_PBMAP_SCHEMA, TUMORSEG_SCHEMA


def predict_tumor_growth(preop_dir: Path, model_id: str, cuda_device: Optional[str] = "0", outdir: Optional[Path] = None) -> None:
    """
    Predict tumor cell concentration from a preprocessed exam using model_id as growth model.

    Parameters:
        preop_dir (Path): Directory to the preoperative exam that has been preprocessed. Should contain the folder with the output.
        model_id (str): Identifier for the model. Used to load the model.
        cuda_device (str): GPU device to use.
        outdir (optional, Path): Base directory for the model output

    Returns:
        None
    """
    logger.info(f"Starting growth prediction on {preop_dir} with {model_id}.")

    model_kwargs = {
            "t1c": MODALITY_STRIPPED_SCHEMA.format(base_dir=preop_dir, modality="t1c"),
            "gm": TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="gm"),
            "wm": TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="wm"),
            "csf": TISSUE_PBMAP_SCHEMA.format(base_dir=preop_dir, tissue="csf"),
            #"gm": TISSUE_SCHEMA.format(base_dir=preop_dir, tissue="gm"),
            #"wm": TISSUE_SCHEMA.format(base_dir=preop_dir, tissue="wm"),
            #"csf": TISSUE_SCHEMA.format(base_dir=preop_dir, tissue="csf"),
            "tumorseg": TUMORSEG_SCHEMA.format(base_dir=preop_dir),
            "outdir": outdir if outdir is not None else preop_dir
            }

    model = TumorGrowthModel(algorithm=model_id, cuda_device=cuda_device)
    model.predict_single(**model_kwargs)


if __name__ == "__main__":
    # Example:
    # python gbm_bench/prediction/predict.py -preop_dir test_data/exam1 -cuda_device 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-cuda_device", type=str, default="0", help="GPU id to run on.")
    parser.add_argument("-preop_dir", type=str, help="Path.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    outdir_tmp = Path("tmp_prediction/")

    predict_tumor_growth(
            preop_dir=Path(args.preop_dir),
            model_id="test_model",
            cuda_device=args.cuda_device,
            outdir=outdir_tmp
            )
