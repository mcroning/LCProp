#!/bin/bash
#SBATCH --job-name=lcprop-pr-2d-opt-s3-test
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=h200-141G
#SBATCH --output=/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-pr-2d-opt-s3-test-%j.stdout
#SBATCH --error=/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-pr-2d-opt-s3-test-%j.stderr

set -u
set -o pipefail
module load cuda/12.9.0

RUN_ROOT=/cluster/tufts/cglab/mcroning/lcprop_runs
SOURCE_ROOT=${RUN_ROOT}/lcprop-pr-2d-opt-stage03-source
PYTHON=/cluster/tufts/cglab/mcroning/condaenv/prenv/bin/python
EXPECTED_SHA=8b5b303bf493d773d7a6f53c4e9354400b35a43f
RUN_DIR=${RUN_ROOT}/pr-2d-gpu-optimization-stage03-validation-${SLURM_JOB_ID}

export PYTHONPATH=${SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/launch.sbatch"

CHECKSUM_FILE=${RUN_DIR}/source_checksums.sha256
cat > "${CHECKSUM_FILE}" <<'EOF'
e00ed4c9d9ef484252150ab98ed4ecb62084ff76631bb9806893e4aeb9e4b4ea  src/lcprop/pr/transverse/static.py
d40df4be8f8cdf1168dd25b10fc7cac686e3e2b460244bff21d961d8ffbf5dd3  tests/test_pr_transverse_static.py
0b3faf028ffad5b82748728c44fb9c7fdcf3d72b9bb685c101454793cfe95b94  src/lcprop/pr/workflow.py
1d7f39480c7ed1d4d248d6c909b48e4b40afd215d4cf316afbe3aca23a3d4fae  src/lcprop/pr/transverse/workflow.py
e85ff777d503a84a5e24cbed7e83018a0dccce5dad964eebabe1c4ef5449cdd4  src/lcprop/pr/transverse/static_workflow.py
7a869d91b97f85acfac83a298d683d920a43839918ca7ffa1a5e785e76a94315  tests/test_pr_scattering.py
1881e361ec9b92df3b22f39c1dc0eb77f7cdeff91c326c65ec9fa17392a6566a  tests/test_pr_transverse_production.py
8f9402d34eabd700c6d4734232b774cdf98d5bcbf022cb9eb13de9aa297a3b3c  tests/test_pr_transverse_static_workflow.py
EOF

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "source_root=${SOURCE_ROOT}"
    echo "expected_git_sha=${EXPECTED_SHA}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    module list 2>&1
    "${PYTHON}" --version
    "${PYTHON}" -c 'import cupy,lcprop; d=cupy.cuda.Device(); p=cupy.cuda.runtime.getDeviceProperties(d.id); n=p["name"]; print("lcprop="+lcprop.__file__); print("cupy="+cupy.__version__); print("cuda_runtime="+str(cupy.cuda.runtime.runtimeGetVersion())); print("device="+(n.decode() if isinstance(n,bytes) else str(n)))'
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    sha256sum "$0"
} > "${RUN_DIR}/provenance.txt"

cd "${SOURCE_ROOT}"
if ! sha256sum --check "${CHECKSUM_FILE}" >> "${RUN_DIR}/provenance.txt" 2>&1; then
    echo "source_gate=failed" >> "${RUN_DIR}/provenance.txt"
    echo 97 > "${RUN_DIR}/exit_code.txt"
    exit 97
fi
echo "source_gate=passed" >> "${RUN_DIR}/provenance.txt"

set +e
"${PYTHON}" -m pytest -q -rs \
    tests/test_splitstep.py \
    tests/test_lc_propagation.py \
    tests/test_pr_midpoint_intensity_reuse.py \
    tests/test_pr_scattering.py \
    tests/test_pr_products.py \
    tests/test_pr_transverse_production.py \
    tests/test_pr_transverse_static_workflow.py \
    tests/test_pr_transverse_static.py \
    tests/test_pr_transverse_static_gpu.py \
    tests/test_pr_transverse_gpu.py \
    --junitxml="${RUN_DIR}/pytest.xml" \
    > "${RUN_DIR}/pytest.stdout" 2> "${RUN_DIR}/pytest.stderr"
TEST_EXIT=$?
set -e
echo "${TEST_EXIT}" > "${RUN_DIR}/exit_code.txt"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RUN_DIR}/provenance.txt"
find "${RUN_DIR}" -maxdepth 1 -type f ! -name artifact_checksums.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "${RUN_DIR}/artifact_checksums.sha256"
exit "${TEST_EXIT}"
