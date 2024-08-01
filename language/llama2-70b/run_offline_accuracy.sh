API_HOST="http://llama-2-70b-chat-hf-isvc-predictor.llama.svc.cluster.local:8080"
CHECKPOINT_PATH=/workspace/llama-model-info/
DATASET_PATH=/workspace/processed-data.pkl
OUTPUT_LOG_DIR="accuracy-logs-$(date +%s)"
ADDITIONAL_SERVERS="http://llama-2-70b-chat-hf-isvc-2-predictor.llama.svc.cluster.local:8080"

python3 -u main.py --scenario Offline --model-path ${CHECKPOINT_PATH} --api-server ${API_HOST} --additional-servers ${ADDITIONAL_SERVERS} --api-model-name Llama-2-70b-chat-hf --mlperf-conf mlperf.conf --vllm --user-conf user.conf --total-sample-count 24576 --dataset-path ${DATASET_PATH} --output-log-dir ${OUTPUT_LOG_DIR} --dtype float32 --device cpu --accuracy --batch-size 3072

ACCURACY_LOG_FILE=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json
if [ -e ${ACCURACY_LOG_FILE} ]; then
        python3 evaluate-accuracy.py --checkpoint-path ${CHECKPOINT_PATH} \
                --mlperf-accuracy-file ${ACCURACY_LOG_FILE} --dataset-file ${DATASET_PATH} --dtype int32
fi
