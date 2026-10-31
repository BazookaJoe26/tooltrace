# ToolTrace

Analyze photographs of objects on specially-marked paper and generate mm-accurate vector traces for CAD/CAM applications.

## Project Status

🚀 **Production Ready** - Full Pipeline Complete

Currently implemented:
- ✓ Project infrastructure and workspace setup
- ✓ CLI argument parsing for both tools
- ✓ **Paper generator with real AprilTag markers**
- ✓ **Unique tag IDs for automatic paper size detection**
- ✓ **Calibration grid and ruler markings**
- ✓ **AprilTag detection and perspective correction**
- ✓ **FastSAM-based object segmentation (GPU accelerated)**
- ✓ **SVG and DXF vector export**
- ✓ **Nested contour removal and false positive filtering**

## Overview

ToolTrace consists of two command-line tools:

### 1. `paper-gen` - Calibration Paper Generator

Generates printable PDFs with AprilTag fiducial markers and calibration grids.

**Usage:**
```bash
# Generate A4 calibration paper
cargo run --bin paper-gen

# Include X and Y axes between the tags to verify print scaling
cargo run --bin paper-gen -- --measurement-axes

# Generate US Letter with custom tag size
cargo run --bin paper-gen -- --size letter --tag-size 60

# All options
paper-gen [OPTIONS]
  -o, --output <FILE>        Output PDF path [default: calibration_paper.pdf]
  -s, --size <SIZE>          Paper size: a4, letter, a3 [default: a4]
  -t, --tag-size <MM>        Outer black square size in millimeters [default: 40.0]
      --measurement-axes   Add X/Y axes between tags with 10 mm ticks [default: off]
```

The default black square measures 40 × 40 mm; its white quiet zone adds 5 mm
on each side. The black square starts 15 mm from the paper edges. Print at actual
size (100%), without fitting to the page. Use the same `--tag-size` value for
`paper-gen` and `tooltrace`.

With `--measurement-axes`, the X axis runs between the centers of the lower
tags and the Y axis between the centers of the left tags. Each line starts and
ends 10 mm from its adjacent black tag edges. Labels are numbers only: 0, every
50 mm, and the endpoint (80 mm in X and 167 mm in Y for A4 with 40 mm tags).
Measure from each axis's 0 tick to its endpoint to check print scaling.
Intermediate ticks are 10 mm apart. The axes are omitted unless the flag is supplied.

**Features:**
- ✓ Supports A4 (210×297mm), US Letter (8.5×11in), and A3 (297×420mm)
- ✓ AprilTag 36h11 markers in corners for perspective detection
- ✓ Unique tag IDs per paper size for automatic detection:
  - A4: Tag IDs 0-3
  - US Letter: Tag IDs 4-7
  - A3: Tag IDs 8-11
- ✓ 10mm calibration grid with 1mm subdivisions
- ✓ Precise ruler markings for verification (1mm, 5mm, 10mm ticks)
- ✓ High-quality tag generation (160x160px embedded images)

### 2. `tooltrace` - Object Tracing Tool

Analyzes photos and extracts object outlines as vector files.

**Usage:**
```bash
# Trace object and output both SVG and DXF
cargo run --bin tooltrace -- --input photo.jpg

# Output only SVG
cargo run --bin tooltrace -- --input photo.jpg --format svg --output trace

# Retry with smoothing only if the first pass does not find all four corner tags
cargo run --bin tooltrace -- --input photo.jpg --detection-smoothing

# Optionally limit the resolution used for tag detection
cargo run --bin tooltrace -- --input photo.jpg --detection-max-dimension 1920 --detection-smoothing

# All options
tooltrace --input <IMAGE> [OPTIONS]
  -i, --input <FILE>         Input image file (required)
  -o, --output <NAME>        Output path without extension [default: output]
  -f, --format <FORMAT>      svg, dxf, or both [default: both]
  -d, --debug                Save intermediate visualizations
  -t, --tag-size <MM>        AprilTag size in millimeters [default: 40.0]
      --detection-max-dimension <PX>  Limit longest image side for tag detection [default: native]
      --detection-smoothing           Allow a second detection pass with 1 px smoothing [default: off]
      --fastsam-model <ONNX>          Override automatic FastSAM model discovery
      --no-attachments               Disable recovery of small attached FastSAM parts
```

**Install FastSAM from a local clone (Windows):**

```powershell
.\scripts\setup-fastsam.ps1 -Repository C:\repos\FastSAM
.\target\debug\tooltrace.exe --input photo.jpg
```

Setup creates an isolated export environment under `target/fastsam-env` and
installs the validated FastSAM-x ONNX model in `models/FastSAM-x.onnx`.
ToolTrace automatically searches `models/` beside the executable and in its
parent directories, then the working directory and its parents. It uses DirectML
when available, with CPU fallback. Python is not required during tracing.
An explicit `--fastsam-model` path takes priority, followed by the
`TOOLTRACE_FASTSAM_MODEL` environment variable. See [model setup details](models/README.md)
for FastSAM-s, portable deployment, and recorded model hashes/package versions.

The model image uses aspect-preserving padding; boxes and full-resolution masks
are mapped back after removing that padding. Final contour processing uses the
original rectified image. `--debug` saves `_fastsam_masks.jpg` and
`_fastsam_contours.jpg`. If no model is installed, the log explicitly identifies
the edge-based fallback. Finding a model does not guarantee dimensional accuracy;
printed-axis validation and contour review remain necessary.

By default, small FastSAM instances that overlap the original main object are
considered as attachments. A candidate must add meaningful exterior area, overlap
at least 5% of its own mask with the original object, cover at most 5% of the main
object's area, and fit within 20 mm. Compactness, image bounds, AprilTag exclusion,
and a connected final union provide additional checks. Nearby disconnected
objects and chains of neighboring masks are not merged. This recovers separately
segmented knobs/wheels without expanding the whole silhouette. Attachments are
merged after straight-line regularization to preserve their small curves.
`--no-attachments` disables this step; `--debug` saves an `_attachments.jpg`
overlay when parts were recovered. Two distinct objects that physically overlap
can still require manual review.

Tag detection uses native resolution by default, including inside the AprilTag
decoder. `--detection-max-dimension` only shrinks images larger than the requested
limit. Detected positions are mapped back to original image coordinates using
the actual resized width and height.

If the decoder misses corner tags, an additional native-resolution contour
search supplies quadrilateral candidates. Their actual 36h11 bit patterns are
decoded and their corners refined against the original detection pixels. This
search does not resize or blur the image and is part of the first detection pass.

With `--detection-smoothing`, the first pass always uses the unblurred image. If
the four corner IDs of a supported paper are incomplete, a second pass uses
Gaussian smoothing (sigma 1 pixel at the selected detection resolution). Missing
IDs are added to the first results; existing marker positions are preserved.
These options only preprocess the image used for detection; perspective
correction and contour extraction continue using the original image.

Object selection prioritizes foreground area and position rather than circularity,
so elongated tools are retained instead of small printed scale marks. When using
the edge-based fallback, the selected outline seeds foreground/background
refinement at native pixel resolution. This separates interior printing from
the outer silhouette. `--debug` also saves a `_refined_contour.jpg` overlay.
Spline controls preserve corners and use bounded pixel spacing to avoid deleting
jaws or bending long straight sides.

Long, sharp image edges also guide shadow-boundary correction. The line detector
works at scale 1.0; matching is restricted to nearby contour runs and less than
half the local object thickness. Short weak gaps can be joined, while strong
notches and rounded features remain separate. Nearly straight runs at least
8 mm long are regularized using a 0.6 mm line-fit tolerance; tangent changes
protect visible arcs. `--debug` saves a `_linearized_contour.jpg` overlay.

DXF export converts image Y-down coordinates to CAD Y-up coordinates so the
outline has the same visible orientation as the photo/SVG. The conversion uses
one shared center for all contours, preserving their relative placement.

Calibration jointly fits all 16 black-square marker corners with a normalized
homography and Huber weighting (0.35 mm threshold in paper coordinates). The
projected marker centers initialize corner assignment; paper IDs determine the
orientation. Physical target coordinates retain fractional pixels; only the
output canvas bounds are rounded. The console reports each corner's fitting
residual, the overall RMS/maximum, and the number of downweighted corners.

After warping, ToolTrace automatically checks the optional printed measurement
axes. It detects the axis strokes, both end ticks, and at least 70% of the cross
ticks, then reports measured lengths and deviations. These observations never
rescale the calibration. Missing, obscured, or inconsistent rulers are reported
as **not verified**; deviations over 0.50 mm are marked **outside tolerance**.
Ruler positions and lengths follow the detected paper size and `--tag-size`.
Results are saved to `<output>_calibration.txt`; `--debug` also writes
`<output>_measurement_axes.jpg` and annotated marker-corner images.

Fitting residuals are not independent validation. Remaining mismatches can
indicate image distortion, a curved sheet, or inaccurate corner detection. The
ruler check concerns the paper plane; objects above that plane can still appear
enlarged, and physical print dimensions must be checked separately. The 300 DPI
output setting alone does not guarantee measurement accuracy.

**Implemented Features:**
- ✓ AprilTag detection for perspective correction (OpenCV + apriltag-rust)
- ✓ Automatic pixel-to-mm calibration (300 DPI output)
- ✓ FastSAM-based object segmentation (GPU accelerated with ONNX Runtime + DirectML)
- ✓ Mask-based contour extraction for precise object outlines
- ✓ Nested contour removal and false positive filtering
- ✓ AprilTag region exclusion
- ✓ SVG and DXF export for Fusion 360
- ✓ Debug visualizations (masks, contours, flattened images)

## Architecture

```
tooltrace/
├── paper-gen/          # PDF generator binary
│   ├── src/
│   │   ├── main.rs
│   │   ├── pdf_generator.rs
│   │   ├── marker_placement.rs
│   │   └── paper_sizes.rs
│   └── Cargo.toml
│
├── tooltrace/          # Image analysis binary
│   ├── src/
│   │   ├── main.rs
│   │   ├── detection.rs      # AprilTag detection
│   │   ├── calibration.rs    # Perspective correction
│   │   ├── segmentation.rs   # Object extraction
│   │   ├── tracing.rs        # Contour tracing
│   │   ├── export_svg.rs     # SVG export
│   │   └── export_dxf.rs     # DXF export
│   └── Cargo.toml
│
└── tooltrace-common/   # Shared types library
    ├── src/
    │   ├── lib.rs
    │   └── types.rs
    └── Cargo.toml
```

## Technology Stack

- **Language:** Rust 2021 Edition
- **Computer Vision:** OpenCV 4.x (opencv-rust)
- **AprilTag Detection:** apriltag-rust + OpenCV
- **AI/ML Inference:** ONNX Runtime 2.0 with DirectML GPU acceleration
- **Segmentation Model:** FastSAM (Fast Segment Anything Model)
- **Image Processing:** image + imageproc crates
- **PDF Generation:** printpdf
- **Vector Export:** svg + dxf crates
- **CLI:** clap v4 with derive macros

## Building

```bash
# Check all crates compile
cargo check --workspace

# Build both binaries
cargo build --release

# Run tests
cargo test --workspace

# Build documentation
cargo doc --workspace --open
```

Binaries will be in `target/release/`:
- `paper-gen.exe` (or `paper-gen` on Unix)
- `tooltrace.exe` (or `tooltrace` on Unix)

## FastSAM Model Setup

The `tooltrace` tool requires the FastSAM ONNX model for object segmentation:

1. **Download FastSAM checkpoint:**
   ```bash
   # Download FastSAM-s.pt from https://github.com/CASIA-IVA-Lab/FastSAM
   # Save to d:/data/FastSAM-s.pt (Windows) or adjust path as needed
   ```

2. **Convert to ONNX format:**
   ```bash
   # Install ultralytics in Python environment
   pip install torch ultralytics onnx

   # Run conversion script
   python convert_fastsam.py
   ```

   Example conversion script:
   ```python
   from ultralytics import YOLO
   model = YOLO("d:/data/FastSAM-s.pt")
   model.export(
       format="onnx",
       imgsz=1024,
       simplify=True,
       dynamic=False,
       opset=12,
   )
   ```

3. **Place ONNX model:**
   - Save the exported `FastSAM-s.onnx` to `d:/data/FastSAM-s.onnx`
   - Or update the path in `tooltrace/src/segmentation.rs:24`

**Fallback:** If FastSAM model is not found, tooltrace falls back to edge-based segmentation (lower quality).

## Development Workflow

1. **Print calibration paper:**
   ```bash
   cargo run --bin paper-gen -- --output cal.pdf
   # Print cal.pdf at actual size (no scaling!)
   ```

2. **Take photo:**
   - Place object on calibration paper
   - Ensure all 4 AprilTag markers are visible
   - Photo can be at an angle (up to ~45°)
   - Use good lighting, avoid shadows

3. **Trace object:**
   ```bash
   cargo run --bin tooltrace -- --input photo.jpg --output part
   # Generates part.svg and part.dxf
   ```

4. **Import to Fusion 360:**
   - Open Fusion 360
   - Insert → Insert DXF
   - Select `part.dxf`
   - Extrude or use as sketch

## Success Criteria

- ±1mm accuracy for objects 50-300mm in size
- Works with smartphone camera photos
- Handles perspective angles up to 45°
- SVG and DXF outputs import correctly into Fusion 360
- Print-accurate PDF generation

## Implementation Progress

See [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md) for detailed development history.

**Current Status:** ✅ Production Ready - Full pipeline operational!

### Recent Milestones
- ✅ FastSAM ONNX model integration with GPU acceleration
- ✅ Mask-based contour extraction (not just bounding boxes)
- ✅ False positive filtering with tunable parameters
- ✅ Nested contour removal algorithm
- ✅ Complete perspective correction pipeline
- ✅ SVG and DXF export ready for Fusion 360

### Performance
- **Segmentation:** ~5 seconds per image (GPU accelerated)
- **Accuracy:** Sub-millimeter precision with proper calibration
- **Memory:** ~1-2 GB during inference
- **Output:** Clean vector contours following actual object shapes

## License

MIT OR Apache-2.0

## Contributing

This project is under active development. Contributions welcome once core functionality is stable.
