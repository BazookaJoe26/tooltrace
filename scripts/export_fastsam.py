"""Export the user's local FastSAM clone for native ToolTrace inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository, weights = args.repository.resolve(), args.weights.resolve()
    if not (repository / "fastsam/model.py").is_file():
        parser.error(f"Not a FastSAM repository: {repository}")
    if not weights.is_file():
        parser.error(f"Missing model checkpoint: {weights}")
    project = Path(__file__).resolve().parent.parent
    config = project / "target/fastsam-config"
    config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config)
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["MPLBACKEND"] = "Agg"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repository))
    from fastsam import FastSAM
    import onnx
    import torch

    torch.set_num_threads(4)
    model = FastSAM(str(weights))
    exported = model.export(format="onnx", imgsz=1024, opset=17,
                            simplify=False, dynamic=False, device="cpu")
    if not exported:
        raise RuntimeError("FastSAM ONNX export failed")
    exported = Path(exported)
    graph = onnx.load(str(exported))
    onnx.checker.check_model(graph)
    shape = lambda value: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
    inputs = [shape(value) for value in graph.graph.input]
    outputs = [shape(value) for value in graph.graph.output]
    if inputs != [[1, 3, 1024, 1024]] or len(outputs) != 2 or outputs[0][:2] != [1, 37] or outputs[1][:2] != [1, 32]:
        raise RuntimeError(f"Unexpected FastSAM tensors: {inputs} -> {outputs}")
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".onnx.partial")
    shutil.copyfile(exported, temporary)
    temporary.replace(destination)
    metadata = dict(model=weights.stem, checkpoint_sha256=digest(weights),
                    onnx_sha256=digest(destination), inputs=inputs, outputs=outputs,
                    exporter_repository=str(repository), torch_version=torch.__version__,
                    onnx_version=onnx.__version__, opset=17,
                    source_url=f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{weights.name}")
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Validated FastSAM model installed: {destination}")
    print(f"Inputs: {inputs}; outputs: {outputs}")


if __name__ == "__main__":
    main()
