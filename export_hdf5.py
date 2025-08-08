"""Convert Procrustes analysis output to a single HDF5 file.

This script collects all rotated binary masks (``frame_*.dat``) and the
kinematics data saved by ``main.py`` (Angle, mass center positions, distance,
surface) and writes them into a single HDF5 file.  This makes the
downstream consumption of the preprocessed video data much easier for
subsequent stages such as midline reconstruction and 3D deformation.

Usage:
    python export_hdf5.py --input-folder <path/to/tmp_results/test_video> \
        --output <path/to/aligned.h5>

The resulting HDF5 file will contain datasets:
    /masks               : (n_frames, height, width) uint8 array of binary masks
    /rigid/angle         : (n_frames,) float array of smoothed angles (degrees)
    /mass_center         : (n_frames, 2) float array of mass centre positions
    /distance            : (n_frames,) float array of cumulative mass centre distances
    /surface             : (n_frames,) float array of segmented fish surface area

and a group ``/meta`` where you can store additional attributes such as
``fps``, ``px_per_mm`` and ``fish_length_mm`` by passing ``--meta``.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional, Dict, Any, List

import numpy as np

try:
    import h5py
except ImportError as e:
    raise ImportError("h5py is required to write the HDF5 file; install it via pip")


def export_procrustes_to_hdf5(
    input_folder: str,
    output_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Aggregate .dat outputs from the Procrustes analysis into an HDF5 file.

    Parameters
    ----------
    input_folder : str
        Path to the directory produced by ``main.py`` containing the rotated frames
        (``frame_XXXXX.dat``) and the ``kinematics_data`` subfolder.
    output_path : str
        Path of the output HDF5 file.  Parent directories will be created if
        necessary.
    meta : dict, optional
        Additional metadata to store under the ``/meta`` group.  Keys and values
        must be JSON serialisable types.
    """
    # Discover mask files
    mask_files: List[str] = sorted(
        glob.glob(os.path.join(input_folder, "frame_*.dat"))
    )
    if not mask_files:
        raise FileNotFoundError(
            f"No frame_*.dat files found in {input_folder}; make sure you pass the correct directory."
        )

    # Load masks into a list of arrays
    masks = [np.loadtxt(f, dtype=np.uint8) for f in mask_files]
    masks_arr = np.stack(masks, axis=0)

    # Load kinematic data
    kin_dir = os.path.join(input_folder, "kinematics_data")
    angle_path = os.path.join(kin_dir, "Angle.dat")
    mass_center_path = os.path.join(kin_dir, "mass_center_pos.dat")
    distance_path = os.path.join(kin_dir, "Distance.dat")
    surface_path = os.path.join(kin_dir, "Surface.dat")

    angles = np.loadtxt(angle_path) if os.path.exists(angle_path) else None
    mass_centers = np.loadtxt(mass_center_path) if os.path.exists(mass_center_path) else None
    distances = np.loadtxt(distance_path) if os.path.exists(distance_path) else None
    surfaces = np.loadtxt(surface_path) if os.path.exists(surface_path) else None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("masks", data=masks_arr, compression="gzip")
        if angles is not None:
            h5f.create_dataset("rigid/angle", data=angles)
        if mass_centers is not None:
            h5f.create_dataset("mass_center", data=mass_centers)
        if distances is not None:
            h5f.create_dataset("distance", data=distances)
        if surfaces is not None:
            h5f.create_dataset("surface", data=surfaces)

        meta_grp = h5f.create_group("meta")
        if meta:
            for key, value in meta.items():
                meta_grp.attrs[key] = value

    print(f"Exported {len(masks)} masks and kinematic data to {output_path}")


def _parse_meta(meta_str: str) -> Dict[str, Any]:
    """Parse a JSON-like string into a dictionary for metadata."""
    import json
    return json.loads(meta_str)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Directory containing frame_*.dat and kinematics_data subfolder",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination HDF5 file",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Optional JSON string with metadata attributes to store under /meta",
    )
    args = parser.parse_args(argv)
    meta = _parse_meta(args.meta) if args.meta else None
    export_procrustes_to_hdf5(args.input_folder, args.output, meta)


if __name__ == "__main__":
    main()