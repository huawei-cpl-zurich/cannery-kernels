# Cannery private runtime

Install `cannery-kernels-runtime` from this directory into an isolated A5 worker
with its matching PyTorch, torch-npu and CANN runtime already provisioned.
This small distribution has no Hub client, dependency downloader or Rust extension.

```python
from cannery_kernels import load_cannery_package

kernel = load_cannery_package(package_bytes, pinned_sha256, trust_remote_code=True)
y = kernel.forward(x)
```

`package_bytes` must be obtained and checksum-verified by the trusted coordinator.
Loading executes native code: only use this API inside a restricted job container,
with a healthy leased device and resource/deadline limits. Keep the loaded kernel
alive until its asynchronous calls finish. No package code runs in the API process.

The initial format is an uncompressed tar containing exactly `metadata.json` and
`kernel.so`, at most64 MiB total. Metadata schema1 declares device`a5`, ABI
`unary-f32-v1`, source SHA256, library SHA256, compiler identity, and exact runtime
fingerprint: machine, Python major/minor, torch, torch-npu, CANN and SoC. Incompatible
packages fail before `dlopen`. No source files or executable Python wrappers are
accepted. The library exports `cannery_launch(stream,input,output,uint64_count)`.

`forward` accepts float32 NPU tensors, preserves shape/dtype/device, materializes
strided inputs and skips launches for empty tensors. Work uses the current NPU
stream; consumers synchronize before inspecting errors or measuring execution.
M3 demonstrates admission and a smoke call, without benchmark or correctness-suite
claims. There is no automatic compilation, Hub lookup or missing-dependency fetch.

Run the standalone CPU protocol suite with:

```sh
python -m unittest discover -s tests/cannery -v
```

In the Cannery fork, the `Cannery private runtime` workflow runs CPU tests,
format/lint checks, wheel/sdist builds and installed-package import checks on
Python3.12/3.13. Shared Python lints remain enabled. Inherited GPU/Nix/release
workflows run only in `huggingface/kernels`, where their runner groups and release
credentials exist. They do not constitute Cannery A5 validation. Real A5 evidence
must be collected explicitly on the configured worker and recorded with revisions,
runtime identity, device health and retained remote handles.

## Static CATLASS packages (runtime 0.2+)

Schema 2 / `catlass-unary-f32-v1` packages contain the same two files, with an
explicit `specializations` array of `{shape: [...], dtype: "float32"}` records.
One to four distinct shapes, one to four dimensions and at most 1,048,576 elements
per shape are admitted. At least one variant must be nonempty. Inputs outside the
declared shape/dtype domain are rejected; empty admitted inputs skip native calls.
The native library embeds AIV objects and returns checked AscendCL statuses.

This path imports no CATLASS, executes no author Python and invokes no compiler or
JIT. The exact pinned CANN/PyTorch NPU runtime is still required. A loaded CATLASS
object is bound to the current NPU device; load a separate object after switching
devices or resetting the ACL context. The native bridge enforces a single-context
lifetime. Launch errors propagate after recording in-flight tensor storage on the
stream. Run untrusted binaries only in restricted job containers. Schema 1 Ascend C
packages retain their original void-returning launch contract and dynamic shapes.
