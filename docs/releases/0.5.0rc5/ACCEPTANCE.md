# Exact 0.5.0rc5 CPU acceptance — 2026-11-20

PASS for fresh Colab CPU opening and saved-result handling. Candidate source
e81905b9e407191e0e777bdf41b986b9483b9503 from merged main91a2fb7.

## Exact artifacts

- `lcprop-0.5.0rc5-py3-none-any.whl`: `5bdc6327a9418b12dab9f7f61f5e74bde108885b5d0c541df4547c980645a4e0`
- `LCProp-PR-workbench-setup.zip`: `c7d93313ec5c9f1679ba45fd6bb9dff00854b578cffc8f7af5676ee8c7549811`
- `LCProp-PR-workbench-0.5.0rc5-candidate.ipynb`: `d0ff898cd1c0e6c7a71eb47d3ff02ecb1fe4ff1c1ba39840e1a046b45549fad9`

## Actual acceptance

Fresh isolated Colab CPU, user-authorized installation of the matching setup ZIP:
verified kit → install → one requested session restart → same Open cell → visible
workbench. Installed distribution 0.5.0rc5 and all LCProp runtime bytes matched the
pinned wheel. No NVIDIA device found and no propagation dispatched.

Runtime: Python 3.13.18, ipywidgets 7.7.1, ipympl 0.10.0, Matplotlib 3.11.2,
NumPy 2.4.6, SciPy 1.16.3. Saved PR result reopened with original T4 provenance,
three completed steps and an editable CPU draft. Input and Physical YZ plots
rendered, including the normalized-field explanation and horizontal colorbar.
Experiment JSON round-trip preserved the request and completed-result object;
reopening the UI retained the saved result.

Original `batch4-pr1000-cupy.zip` SHA256:
`ea18cc394e34dd54d1c4afe21baff88205098b83605678963269e645ba31be57`.
All nine re-exported member hashes matched exactly:

| Member | SHA256 |
| --- | --- |
| `output/RESULT_READY.json` | `ddc0670309f6cdb59e5c288c601c894849551843aaf9cfeab42ecfd9c6e8b5c2` |
| `output/checksums.sha256` | `1e0c51ebb46ad24699ce72a2b8587c47e344c881208644a8f4052c30c2ad8f91` |
| `output/manifest.json` | `693cf66e582b3a0821fb1d96a7a003e4cb0deb0e9c0490953dbdcd29fdff792d` |
| `output/result.json` | `0d4a611117db174e1adc70c9bd91bd4ac33700b54f5945f3585074c9736669ce` |
| `output/result_arrays.npz` | `3d9ef68cdb4f8f67d2d95bcf49299bcfc3b11163d3d7c7d9f87f14b69e93da0f` |
| `request/READY.json` | `d35412687255b6f92206a64e79bb03e7cf5ce52b6632dbd773b1a1e9d65f39e2` |
| `request/checksums.sha256` | `83f70eed3dc3ff6a2b7cf830e24fefb65145dcfe0e6792cba494c3de671352f8` |
| `request/manifest.json` | `702a7a5fcf3a9daec51c58279844543aa0bc118b65f7ca75e98f639b58ea8129` |
| `request/request.json` | `164aa189afd6259ed63d3c5cf4bdbcb22f8ec312d08b2f149411a9a30b70d7b6` |

The first reopen was attempted before the large sidebar upload completed and
raised BadZipFile. The error was preserved. After upload completion, the original
archive hash and retry passed. No software/data repair or request change was used.
Wait for sidebar upload completion before choosing Open selected file.

Acceptance notebook: https://colab.research.google.com/drive/1svK3litEETJl7LxQ0wrsIndPXgUHnYyB
Detailed local receipt, DOM and screenshot records remain in the project handoff's
`pr-workbench-cpu-rc5-records-2026-11-20/hosted-cpu-acceptance` directory. The notebook
is account-owned and is not a public download prerequisite.

## Evidence boundaries

Earlier frontend evidence at 37eba26 (development wheel a782d0fe/setup ef4d3a9d):
111 local tests passed, 5 skipped; focused review 40 passed. Fresh development-kit
CPU checks covered beam precision, automatic file discovery, no GPU probe for CPU
help, normalized plots and same-controller reopening. Those are prior code checks,
not reruns on rc5. rc5 runtime bytes are identical to that accepted code.

Exact rc5 acceptance above is a separate actual install/file/UI check. It is not
new numerical qualification, GPU acceptance, a fresh hosted Run/Stop test or proof
of recovery after runtime deletion. Physically thin slices retain white space.
No propagation or GPU calculations occurred. CPU default, reduced-only interface,
public memory/ZIP/time limits and existing device restrictions remain unchanged.
Desktop and published earlier releases remain intact.
