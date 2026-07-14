#!/bin/bash
# End-to-End Pepper Dataset Regeneration & Training Pipeline
# Run Name: industrial_v2_800_scenes

REPO_ROOT="/home/user/menon/workspace/software/pepper_scene_graphs"
DATA_ROOT="/home/data/menon/data_pepper_scenes"
RUN_NAME="industrial_v2_800_scenes_$(date +%Y%m%d_%H%M)"
OUTPUT_DIR="${DATA_ROOT}/${RUN_NAME}"
LOG_FILE="${REPO_ROOT}/${RUN_NAME}_pipeline.log"

export PYTHONPATH=$PYTHONPATH:${REPO_ROOT}

echo "[$(date)] Starting Pipeline: ${RUN_NAME}" | tee -a ${LOG_FILE}

# 1. Parallel Scene Generation (800 Scenes)
echo "[$(date)] Phase 1: Generating 800 scenes with 16 workers..." | tee -a ${LOG_FILE}
/home/user/menon/anaconda3/envs/pepper_train/bin/python ${REPO_ROOT}/scripts/dataset_generation/run_chunked_generation.py \
    --scene /home/data/menon/data_pepper_scenes/template/building_block_models.blend \
    --scenes-root ${DATA_ROOT} \
    --run-name "industrial_v2_800" \
    --total-scenes 800 \
    --chunk-size 25 \
    --workers 16 \
    --bproc-bin /home/user/menon/anaconda3/envs/bproc/bin/blenderproc \
    --extract-pc \
    --voxel-size 0.002 >> ${LOG_FILE} 2>&1

GEN_EXIT=$?
if [ $GEN_EXIT -ne 0 ]; then
    echo "ERROR: Generation failed with exit code $GEN_EXIT. Aborting training." | tee -a ${LOG_FILE}
    exit 1
fi

# 2. Identify the actual output directory (since run_chunked_generation appends a timestamp)
LATEST_RUN=$(ls -td ${DATA_ROOT}/industrial_v2_800_* | head -n 1)
ACTUAL_DATA_DIR="${LATEST_RUN}/data"

echo "[$(date)] Generation Complete. Data is at: ${ACTUAL_DATA_DIR}" | tee -a ${LOG_FILE}

# 3. Verify Dataset Readiness
echo "[$(date)] Phase 2: Verifying dataset integrity (800 scenes)..." | tee -a ${LOG_FILE}
/home/user/menon/anaconda3/envs/pepper_train/bin/python ${REPO_ROOT}/scripts/analysis/verify_dataset_v2.py ${LATEST_RUN} --count 800 >> ${LOG_FILE} 2>&1

VERIFY_EXIT=$?
if [ $VERIFY_EXIT -ne 0 ]; then
    echo "ERROR: Dataset verification failed. Check ${LOG_FILE} for details. Aborting training." | tee -a ${LOG_FILE}
    exit 1
fi

# 4. Success Pause
echo "[$(date)] Verification PASSED. Pausing for 10 minutes before starting training..." | tee -a ${LOG_FILE}
sleep 600

# 5. Start PointNet++ Training
echo "[$(date)] Phase 3: Launching training..." | tee -a ${LOG_FILE}
export PEPPER_DATA_ROOT="${ACTUAL_DATA_DIR}"
/home/user/menon/anaconda3/envs/pepper_train/bin/python ${REPO_ROOT}/pepper_scene_graphs/training/train_semseg.py >> ${LOG_FILE} 2>&1

echo "[$(date)] Pipeline Finished." | tee -a ${LOG_FILE}
