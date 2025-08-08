# Enhanced Zebrafish Procrustes Analysis

This project provides tools for analyzing zebrafish movement using Procrustes analysis on video data. It includes enhanced functionality to collect all rotated binary frames and kinematic data into a single HDF5 file for easier data processing and analysis.

## Features

- Video preprocessing and frame analysis
- Procrustes analysis for zebrafish movement
- HDF5 export for consolidated data storage
- GUI-based smoothing application
- Automated testing and code quality tools

## Installation

This project uses modern Python packaging. Install it in development mode:

```bash
pip install -e .
```

Or install with additional development dependencies:

```bash
pip install -e ".[test,format,lint]"
```

## Development Tools

This project includes several development utilities:

### Code Formatting and Linting

Format and lint the code:

```bash
tox -e format
```

Check formatting and linting without making changes:

```bash
tox -e lint
```

### Testing

Run tests:

```bash
pytest
```

## Usage

Run the main analysis:

```bash
python main.py
```

Or use the console script (after installation):

```bash
zebrafish-analysis
```

## Contents

* **patch.diff** – A unified diff showing the changes to the original
  ``video_preprocessing/experiment.py`` and ``main.py``.  It introduces
  optional parameters to ``process_frames_from_smoothed_angle`` to accumulate
  rotated frames and write them to an HDF5 file alongside the smoothed angles
  and mass centres.  In ``main.py`` the call to this method is amended to
  provide an ``aligned.h5`` path.
* **export_hdf5.py** – A standalone script for converting the existing
  ``.dat`` outputs (``frame_*.dat`` and the kinematics data in
  ``kinematics_data``) into a consolidated HDF5 file.  This is useful if you
  do not want to modify the original code but still need the HDF5 format.

## Applying the patch

To apply the patch to your local clone of
``TheoMrc/zebrafish-procrustes-analysis``, run:

```bash
git apply path/to/enhanced_procrustes_analysis/patch.diff
```

This will update ``video_preprocessing/experiment.py`` and ``main.py`` to
include HDF5 export support.  After applying the patch, the pipeline will
write ``aligned.h5`` into the target directory alongside the ``.dat`` files.

Alternatively, you can keep your current code unchanged and run
``export_hdf5.py`` after processing to bundle the results:

```bash
python export_hdf5.py --input-folder test/tmp_results/test_video --output test/tmp_results/test_video/aligned.h5
```

## HDF5 format

The produced HDF5 file has the following structure:

```
/masks               (n_frames, 300, 300) uint8 – aligned binary masks
/rigid/angle         (n_frames,) float    – smoothed rotation angles
/mass_center         (n_frames, 2) float  – mass centre positions
/distance            (n_frames,) float    – cumulative travel distance (optional)
/surface             (n_frames,) float    – segmented surface area (optional)
/meta                group with attributes such as fps, px_per_mm, fish_length_mm
```

This layout matches the expectations of the new zebrafish deformation repo,
which reads the masks and rigid motion parameters from a single file and
computes midlines, curvature and widths.
