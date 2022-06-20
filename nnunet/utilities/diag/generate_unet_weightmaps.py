import argparse
import json
from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.ndimage import (
    binary_erosion,
    generate_binary_structure,
    distance_transform_edt,
)


def compute_unet_border_weightmaps(
    seg_input: np.ndarray,
    classes: Tuple[int] = (1, 2, 3, 4, 5),
    sigma: float = 5.0,
    w_0: float = 10.0,
) -> np.ndarray:

    # segmentation input cleanup - ensure only 0 (bg) and classes defined in classes are present in seg
    seg = np.copy(seg_input)
    seg_mask = np.isin(seg, classes)
    seg[~seg_mask] = 0

    # computed segmentation mask and class frequencies
    klass_frequency = {}
    final_mask = np.zeros_like(seg, dtype=bool)
    for klass in classes:
        segmented_klass = seg == klass
        klass_frequency[klass] = np.sum(segmented_klass)
        output_structure = binary_erosion(
            segmented_klass, structure=generate_binary_structure(3, 1)
        )
        final_mask = final_mask | output_structure

    # generate klass_frequencies w_c(x)
    klass_frequency[0] = (seg == 0).sum()
    klass_frequency = {
        k: 1.0 - v / np.prod(seg.shape) for k, v in klass_frequency.items()
    }
    w_c = np.zeros_like(seg, dtype=np.float32)
    for k, v in klass_frequency.items():
        w_c[seg == k] = v

    # compute distance to each nearest class segment in the segmentation mask
    dist_1 = distance_transform_edt(~final_mask)
    dist_2 = 0  # we ignore the second greatest distance

    # compute weightmap
    weight_map = w_c + w_0 * np.exp(
        -(np.power(dist_1 + dist_2, 2)) / (2 * sigma * sigma)
    )

    # weightmap masking with known klass_frequencies...
    m = final_mask > 0
    weight_map[m] = w_c[m]

    return weight_map


def load_segmentation_data(input_file: Path) -> np.ndarray:
    if input_file.suffix == ".npz":
        return np.load(str(input_file))["data"][1, :]
    elif input_file.suffix == ".npy":
        return np.load(str(input_file))[1, :]
    raise NotImplementedError(
        ".npz and .npy are the only supported file formats for now..."
    )


def generate_unet_weightmaps(
    input_dir: Path,
    output_dir: Path,
    matching_pattern: str = "*.npz",
    classes: Tuple = (1, 2, 3, 4, 5),
    sigma: float = 5.0,
    w_0: float = 10.0,
):
    input_files = list(input_dir.glob(matching_pattern))
    for idx, input_file in enumerate(input_files):
        output_file = output_dir / (input_file.stem + "_weightmap.npy")
        print(
            f"Processing {idx + 1:3}/{len(input_files):3} : {input_file} -> {output_file}"
        )
        seg = load_segmentation_data(input_file=input_file)
        wmap = compute_unet_border_weightmaps(
            seg_input=seg, classes=classes, sigma=sigma, w_0=w_0
        )
        wmap = np.expand_dims(wmap, axis=0)  # Add a dummy dimension to the data
        print(f"Outputing data: {wmap.shape} to {output_file}")
        np.save(str(output_file), wmap)


def get_classes_from_dataset_file(input_dir: Path, background_label: int = 0) -> Tuple[int]:
    dataset_file = input_dir.resolve().absolute().parent / "dataset.json"
    if not dataset_file.is_file():
        raise FileNotFoundError(
            f"No dataset file with labels found at: {dataset_file}"
        )
    with open(dataset_file, "r") as f:
        meta_data = json.load(fp=f)
    return tuple([int(k) for k, v in meta_data["labels"].items() if int(k) != background_label])


def generate_unet_weightmaps_cli():
    parser = argparse.ArgumentParser("Generate U-Net weightmaps for nnUNet tool")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory with preprocessed nnUNet data labels (.npz/.npy) files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Output directory to store the generated U-Net weightmaps. "
        "By default this is the same as the input directory.",
    )
    parser.add_argument(
        "--matching-pattern",
        type=str,
        default="*.npz",
        help="Input file matching pattern to create weightmaps for.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Sigma value for computing the weightmap edges. Default: 5.0",
    )
    parser.add_argument(
        "--w0",
        type=float,
        default=10.0,
        help="w_0 value for computing the weightmap edges. Default: 10.0",
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=int,
        nargs="*",
        default=(),
        required=False,
        help="All non-background classes to use for creating the weightmap. "
        "Default: Auto-detect classes with 0 as background class",
    )

    args = parser.parse_args()
    if args.output_dir is None:
        print("No output dir was set, using input dir as output dir...")
        args.output_dir = args.input_dir
    if not any(args.classes):
        print("No classes were specified, attempting to get classes from dataset.json... ")
        print("Background value is assumed to be 0")
        args.classes = get_classes_from_dataset_file(
            input_dir=Path(args.input_dir), background_label=0
        )

    print(
        f"Computing weightmaps for classes: {args.classes} with sigma: {args.sigma} and w_0: {args.w0}"
    )
    generate_unet_weightmaps(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        matching_pattern=args.matching_pattern,
        sigma=args.sigma,
        w_0=args.w0,
        classes=args.classes,
    )


if __name__ == "__main__":
    generate_unet_weightmaps_cli()
