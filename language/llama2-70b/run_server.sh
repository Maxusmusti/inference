API_HOST="http://llama-2-70b-chat-hf-isvc-predictor.llama.svc.cluster.local:8080"
CHECKPOINT_PATH=/workspace/llama-model-info/
DATASET_PATH=/workspace/processed-data.pkl
LOGDIR="offline-logs-$(date +%s)"
ADDITIONAL_SERVERS="http://llama-2-70b-chat-hf-isvc-2-predictor.llama.svc.cluster.local:8080"

python3 -u main.py --scenario Server --model-path ${CHECKPOINT_PATH} --api-server ${API_HOST} --additional-servers ${ADDITIONAL_SERVERS} --api-model-name Llama-2-70b-chat-hf --mlperf-conf mlperf.conf --vllm --user-conf user.conf --total-sample-count 24576 --dataset-path ${DATASET_PATH} --output-log-dir ${LOGDIR} --dtype float32 --device cpu
