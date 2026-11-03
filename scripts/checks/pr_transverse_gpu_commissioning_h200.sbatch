#!/bin/bash
#SBATCH --job-name=lcprop-pr-transverse-gpu
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=h200-141G
#SBATCH --output=/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-pr-transverse-gpu-%j.stdout
#SBATCH --error=/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-pr-transverse-gpu-%j.stderr

set -u
set -o pipefail

module load cuda/12.9.0

SNAPSHOT_DIR=/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-transverse-gpu-50ac739-overlay
RUN_ROOT=/cluster/tufts/cglab/mcroning/lcprop_runs
PYTHON=/cluster/tufts/cglab/mcroning/condaenv/prenv/bin/python
EXPECTED_SHA=50ac739236cd175888ec0cf142fca1afc4464208
RUN_DIR="${RUN_ROOT}/pr-transverse-gpu-${SLURM_JOB_ID}"
PROVENANCE_FILE="${RUN_DIR}/provenance.txt"

export PYTHONPATH="${SNAPSHOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RUN_DIR}"
cd "${SNAPSHOT_DIR}"

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "snapshot=${SNAPSHOT_DIR}"
    echo "snapshot_is_uncommitted_overlay=true"
    echo "expected_base_git_sha=${EXPECTED_SHA}"
    echo "actual_base_git_sha=$(git rev-parse HEAD)"
    echo "pythonpath=${PYTHONPATH}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    echo "loaded_modules:"
    module list 2>&1
    "${PYTHON}" --version
    "${PYTHON}" -c 'import cupy, lcprop; print(f"lcprop={lcprop.__file__}"); print(f"cupy={cupy.__version__}"); print(f"cuda_runtime={cupy.cuda.runtime.runtimeGetVersion()}"); d=cupy.cuda.Device(); p=cupy.cuda.runtime.getDeviceProperties(d.id); n=p.get("name"); print("device="+(n.decode() if isinstance(n,bytes) else str(n)))'
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    echo "overlay_sha256:"
    sha256sum \
        "${SNAPSHOT_DIR}/src/lcprop/pr/scattering.py" \
        "${SNAPSHOT_DIR}/src/lcprop/pr/transverse/transport.py" \
        "${SNAPSHOT_DIR}/src/lcprop/pr/transverse/diagnostics.py" \
        "${SNAPSHOT_DIR}/src/lcprop/pr/transverse/projection.py" \
        "${SNAPSHOT_DIR}/src/lcprop/pr/transverse/specs.py" \
        "${SNAPSHOT_DIR}/src/lcprop/pr/transverse/workflow.py" \
        "${SNAPSHOT_DIR}/tests/test_pr_scattering.py" \
        "${SNAPSHOT_DIR}/tests/test_pr_td_canonical_scattering.py" \
        "${SNAPSHOT_DIR}/tests/test_pr_transverse_gpu.py" \
        "${SNAPSHOT_DIR}/scripts/checks/pr_transverse_gpu_commissioning.py"
} > "${PROVENANCE_FILE}"

set +e
"${PYTHON}" -m pytest -q -rs \
    tests/test_pr_scattering.py \
    tests/test_pr_td_canonical_scattering.py \
    tests/test_pr_transverse_reference.py \
    tests/test_pr_transverse_imex.py \
    tests/test_pr_transverse_production.py \
    tests/test_pr_transverse_gpu.py \
    --junitxml="${RUN_DIR}/transverse-gpu.xml"
TEST_EXIT=$?
if [ "${TEST_EXIT}" -eq 0 ]; then
    "${PYTHON}" scripts/checks/pr_transverse_gpu_commissioning.py \
        --grids 256 1024 \
        --output "${RUN_DIR}/commissioning.json"
    BENCHMARK_EXIT=$?
else
    BENCHMARK_EXIT=99
fi
set -e

FINAL_EXIT=${TEST_EXIT}
if [ "${FINAL_EXIT}" -eq 0 ] && [ "${BENCHMARK_EXIT}" -ne 0 ]; then
    FINAL_EXIT=${BENCHMARK_EXIT}
fi
{
    echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "test_exit=${TEST_EXIT}"
    echo "benchmark_exit=${BENCHMARK_EXIT}"
    echo "exit_code=${FINAL_EXIT}"
} >> "${PROVENANCE_FILE}"
echo "${FINAL_EXIT}" > "${RUN_DIR}/exit_code.txt"
exit "${FINAL_EXIT}"
