# Using OpenShift AI Model Serving (Caikit/TGIS) with MLPerf Inference

Prerequisites:
 - Install the OpenShift AI model serving stack
 - Add your AWS credentials to `secret.yaml` access the model files
 - Apply `secret.yaml`, `sa.yaml`, `serving-runtime.yaml`, then finally `model.yaml`
 - Create a benchmark pod using `benchmark.yaml`

In the pod, before any benchmark, first run `cd inference/language/llama2-70b`

For the full accuracy benchmark (offline), run in the pod:
```
python3 -u main.py --scenario Offline --model-path ${CHECKPOINT_PATH} --api-server <INSERT SERVER API CALL ENDPOINT> --api-model-name Llama-2-70b-chat-hf-caikit --accuracy --mlperf-conf mlperf.conf --user-conf user.conf --total-sample-count 24576 --dataset-path ${DATASET_PATH} --output-log-dir offline-logs --dtype float32 --device cpu 2>&1 | tee offline_performance_log.log
```
You can then run the same evaluation/consolidation scripts as the regular benchmark


For the performance benchmark (offline), run in the pod:
```
python3 -u main.py --scenario Offline --model-path ${CHECKPOINT_PATH} --api-server <INSERT SERVER API CALL ENDPOINT> --api-model-name Llama-2-70b-chat-hf-caikit --mlperf-conf mlperf.conf --user-conf user.conf --total-sample-count 24576 --dataset-path ${DATASET_PATH} --output-log-dir offline-logs --dtype float32 --device cpu 2>&1 | tee offline_performance_log.log
```
(It is the same, just with `--accuracy` removed)


For the performance benchmark (server), run in the pod:
```
python3 -u main.py --scenario Server --model-path ${CHECKPOINT_PATH} --api-server <INSERT SERVER STREAM API CALL ENDPOINT> --api-model-name Llama-2-70b-chat-hf-caikit --mlperf-conf mlperf.conf --user-conf user.conf --total-sample-count 24576 --dataset-path ${DATASET_PATH} --output-log-dir server-logs --dtype float32 --device cpu 2>&1 | tee server_performance_log.log
```
(Configure target qps in `user.conf`)


NOTE: Hyperparams are currently configured for 8xH100
