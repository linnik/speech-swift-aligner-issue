"""Show what the Neural Engine returns for pow(x, 2), next to mul(x, x).

Run with `uv run ane_pow.py`. It builds a small Core ML model from scratch and needs no download.
"""
import collections
import shutil
import tempfile

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan

# 1. A small model: y = conv(x), then pow(y, 2) and mul(y, y), each followed by a convolution.
#    With the 1x1 convolutions around them, Core ML places both squares on the Neural Engine.
channels, width = 256, 768
identity = np.eye(channels, dtype=np.float32).reshape(channels, channels, 1, 1)
input_spec = mb.TensorSpec(shape=(1, channels, 1, width), dtype=types.fp32)


@mb.program(input_specs=[input_spec], opset_version=ct.target.iOS17)
def squares(x):
    y = mb.conv(x=x, weight=identity)
    power = mb.pow(x=y, y=np.float32(2.0))
    product = mb.mul(x=y, y=y)
    return (
        mb.conv(x=power, weight=identity, name="pow"),
        mb.conv(x=product, weight=identity, name="mul"),
    )


folder = tempfile.mkdtemp()
model = ct.convert(
    squares,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16,
)
model.save(f"{folder}/squares.mlpackage")
compiled = ct.utils.compile_model(f"{folder}/squares.mlpackage", f"{folder}/squares.mlmodelc")

# 2. Where Core ML places each operation when the Neural Engine is allowed.
plan = MLComputePlan.load_from_path(path=compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
placement = collections.Counter()
for operation in plan.model_structure.program.functions["main"].block.operations:
    usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
    if usage is None:  # a constant has no device
        continue
    device = type(usage.preferred_compute_device).__name__
    device = device.removeprefix("ML").removesuffix("ComputeDevice")
    name = operation.operator_name.split(".")[-1]
    placement[f"{name} on {device}"] += 1
print("Placement with CPU_AND_NE:", dict(placement))

# 3. The input: random values of both signs, with a few fixed values first to print.
shown = np.array([-3, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 3], dtype=np.float32)
rng = np.random.default_rng(0)
x = 2 * rng.standard_normal((1, channels, 1, width)).astype(np.float32)
x[0, 0, 0, : len(shown)] = shown
exact = x.astype(np.float64) ** 2
negative = x < 0

# 4. Both squares on each compute unit. The last column is the share of negative inputs
#    for which the result equals x^2.
header = " ".join(f"{value:8g}" for value in shown)
print(f"\n{'x':24} {header}   negative x equal to x^2")
for units in ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"]:
    compute_units = getattr(ct.ComputeUnit, units)
    outputs = ct.models.CompiledMLModel(compiled, compute_units=compute_units).predict({"x": x})
    for name in ["pow", "mul"]:
        result = outputs[name].astype(np.float64)
        row = " ".join(f"{value:8.3g}" for value in result[0, 0, 0, : len(shown)])
        equal = np.isclose(result[negative], exact[negative], rtol=0.02, atol=1e-3).mean()
        label = f"{name}(x) on {units}"
        print(f"{label:24} {row}   {equal:6.1%}")

shutil.rmtree(folder)
