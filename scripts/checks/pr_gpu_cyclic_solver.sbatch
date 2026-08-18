#!/bin/bash
#SBATCH --job-name=lcprop-pr-gpu-cyclic
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --output=/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-pr-gpu-cyclic-%j.stdout
#SBATCH --error=/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-pr-gpu-cyclic-%j.stderr

set -u
set -o pipefail

module load cuda/12.9.0

SNAPSHOT_DIR=/cluster/tufts/cglab/mcroning/lcprop_runs/pr-gpu-cyclic-d22863b-snapshot
RUN_ROOT=/cluster/tufts/cglab/mcroning/lcprop_runs
PYTHON=/cluster/tufts/cglab/mcroning/condaenv/prenv/bin/python
EXPECTED_SHA=d22863b745856846630106b1a1687f48717630f6
RUN_DIR="${RUN_ROOT}/pr-gpu-cyclic-${SLURM_JOB_ID}"
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
    echo "expected_git_sha=${EXPECTED_SHA}"
    echo "pythonpath=${PYTHONPATH}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    echo "loaded_modules:"
    module list 2>&1
    "${PYTHON}" --version
    "${PYTHON}" -c 'import cupy, lcprop; print(f"lcprop={lcprop.__file__}"); print(f"cupy={cupy.__version__}"); d=cupy.cuda.Device(); p=cupy.cuda.runtime.getDeviceProperties(d.id); n=p.get("name"); print("device="+(n.decode() if isinstance(n,bytes) else str(n)))'
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    echo "snapshot_sha256:"
    sha256sum \
        "${SNAPSHOT_DIR}/src/lcprop/pr/cyclic_gpu.py" \
        "${SNAPSHOT_DIR}/src/lcprop/pr/evolution.py" \
        "${SNAPSHOT_DIR}/tests/test_pr_cyclic_gpu.py" \
        "${SNAPSHOT_DIR}/scripts/checks/pr_gpu_cyclic_solver.py"
} > "${PROVENANCE_FILE}"

set +e
"${PYTHON}" -m pytest -q -rs \
    "${SNAPSHOT_DIR}/tests/test_pr_cyclic_gpu.py" \
    tests/test_pr_semi_implicit_prototype.py::test_semi_implicit_prototype_is_cupy_compatible_when_available \
    tests/test_pr_vertical_slice.py::test_pr_primitive_functions_are_cupy_compatible_when_available \
    --junitxml="${RUN_DIR}/gpu-cyclic.xml"
TEST_EXIT=$?
if [ "${TEST_EXIT}" -eq 0 ]; then
    "${PYTHON}" "${SNAPSHOT_DIR}/scripts/checks/pr_gpu_cyclic_solver.py" \
        --output "${RUN_DIR}/benchmark.json" --repeats 3
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
