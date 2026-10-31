# FastSAM models

Run from the ToolTrace repository:

```powershell
.\scripts\setup-fastsam.ps1 -Repository C:\repos\FastSAM
```

This uses the local FastSAM clone, creates an isolated Python environment under
`target/fastsam-env`, installs the README dependencies, downloads the official
FastSAM-x checkpoint, and exports a checked ONNX model to `models/FastSAM-x.onnx`.
The optional CLIP package is unnecessary for ToolTrace's automatic segmentation.
Use `-Variant s` for FastSAM-s, or `-SkipDependencies` to reuse the environment.
Python is used only during setup/export. ToolTrace runs the ONNX model directly.

The generated JSON records tensor dimensions, checkpoint/export SHA-256 hashes,
and the source download URL. `target/fastsam-environment.txt` records the Python
package versions. Model weights and generated metadata are not committed.

Automatic discovery checks `models/` beside the executable and in its parent
directories, then the working directory and its parents. Within each directory,
the preference is FastSAM-x, FastSAM-s, then FastSAM.onnx. For a portable build,
copy the ONNX model into a `models` directory beside `tooltrace.exe`.

To override discovery:

```powershell
.\target\debug\tooltrace.exe --input photo.jpg --fastsam-model C:\models\FastSAM-s.onnx
# Or set TOOLTRACE_FASTSAM_MODEL to the ONNX model path.
```

A configured path that does not exist is an error. If no model is installed,
ToolTrace explicitly reports that it is using the edge-based fallback. A model
that exists but cannot be loaded is reported as an error rather than silently
using a different algorithm.
