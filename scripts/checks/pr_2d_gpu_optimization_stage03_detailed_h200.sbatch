#!/bin/bash
#SBATCH --job-name=lcprop-pr-2d-opt-s3-detail
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=h200-141G
#SBATCH --output=/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-pr-2d-opt-s3-detail-%j.stdout
#SBATCH --error=/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-pr-2d-opt-s3-detail-%j.stderr

set -u
set -o pipefail
module load cuda/12.9.0

RUN_ROOT=/cluster/tufts/cglab/mcroning/lcprop_runs
SOURCE_ROOT=${RUN_ROOT}/lcprop-pr-2d-opt-stage03-source
RUNNER=${RUN_ROOT}/pr_2d_gpu_optimization_stage03.py
STAGE01_RUNNER=${RUN_ROOT}/pr_2d_gpu_optimization_stage01.py
PYTHON=/cluster/tufts/cglab/mcroning/condaenv/prenv/bin/python
EXPECTED_SHA=8b5b303bf493d773d7a6f53c4e9354400b35a43f
EXPECTED_RUNNER_SHA=fa323845381c609095e273a76650243274ec1343349a58e51bd3bfa175e70e00
EXPECTED_STAGE01_RUNNER_SHA=bd9a53b6e47a393e00bb9eeffdd004b69507ad478bd1078e83f942d8ff04e25a
RUN_DIR=${RUN_ROOT}/pr-2d-gpu-optimization-stage03-detailed-${SLURM_JOB_ID}

export PYTHONPATH=${SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}
export LCPROP_PROFILE_SOURCE_SHA=${EXPECTED_SHA}
mkdir -p "${RUN_DIR}"
cp "$0" "${RUN_DIR}/launch.sbatch"
cp "${RUNNER}" "${RUN_DIR}/stage03_runner.py"
cp "${STAGE01_RUNNER}" "${RUN_DIR}/pr_2d_gpu_optimization_stage01.py"

CHECKSUM_FILE=${RUN_DIR}/source_checksums.sha256
cat > "${CHECKSUM_FILE}" <<'EOF'
e00ed4c9d9ef484252150ab98ed4ecb62084ff76631bb9806893e4aeb9e4b4ea  src/lcprop/pr/transverse/static.py
0b3faf028ffad5b82748728c44fb9c7fdcf3d72b9bb685c101454793cfe95b94  src/lcprop/pr/workflow.py
1d7f39480c7ed1d4d248d6c909b48e4b40afd215d4cf316afbe3aca23a3d4fae  src/lcprop/pr/transverse/workflow.py
e85ff777d503a84a5e24cbed7e83018a0dccce5dad964eebabe1c4ef5449cdd4  src/lcprop/pr/transverse/static_workflow.py
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
    sha256sum "${RUNNER}" "${STAGE01_RUNNER}" "$0"
} > "${RUN_DIR}/provenance.txt"

cd "${SOURCE_ROOT}"
ACTUAL_RUNNER_SHA=$(sha256sum "${RUNNER}" | awk '{print $1}')
ACTUAL_STAGE01_RUNNER_SHA=$(sha256sum "${STAGE01_RUNNER}" | awk '{print $1}')
if [ "${ACTUAL_RUNNER_SHA}" != "${EXPECTED_RUNNER_SHA}" ] || \
   [ "${ACTUAL_STAGE01_RUNNER_SHA}" != "${EXPECTED_STAGE01_RUNNER_SHA}" ]; then
    echo "runner_gate=failed" >> "${RUN_DIR}/provenance.txt"
    echo 97 > "${RUN_DIR}/exit_code.txt"
    exit 97
fi
if ! sha256sum --check "${CHECKSUM_FILE}" >> "${RUN_DIR}/provenance.txt" 2>&1; then
    echo "source_gate=failed" >> "${RUN_DIR}/provenance.txt"
    echo 97 > "${RUN_DIR}/exit_code.txt"
    exit 97
fi
echo "runner_gate=passed" >> "${RUN_DIR}/provenance.txt"
echo "source_gate=passed" >> "${RUN_DIR}/provenance.txt"

set +e
"${PYTHON}" "${RUNNER}" --output-dir "${RUN_DIR}/detailed" --mode detailed \
    --sizes 1024 > "${RUN_DIR}/detailed.stdout" 2> "${RUN_DIR}/detailed.stderr"
DETAIL_EXIT=$?
set -e

{
    echo "detailed_exit=${DETAIL_EXIT}"
    echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    du -sh "${RUN_DIR}"
} >> "${RUN_DIR}/provenance.txt"
echo "${DETAIL_EXIT}" > "${RUN_DIR}/exit_code.txt"
find "${RUN_DIR}" -type f ! -name artifact_checksums.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "${RUN_DIR}/artifact_checksums.sha256"
exit "${DETAIL_EXIT}"
