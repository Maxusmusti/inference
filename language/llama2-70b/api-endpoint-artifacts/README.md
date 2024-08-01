
# Deploying Llama-2-70b with vLLM on OpenShift AI, and running MLPerf Inference

## Prerequisites:

### Environment setup:
Our benchmarks were run on OpenShift Container Platform and OpenShift AI's model serving stack. This includes the following setup:
* [Red Hat OpenShift container platform 4.15](https://access.redhat.com/documentation/en-us/openshift_container_platform/4.15/html/installing/index)
*  [Node Feature Discovery Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-01.html)
* [Nvidia GPU Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-03.html)
* [OpenShift AI model serving stack](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.11/html/serving_models/serving-large-models_serving-large-models#configuring-automated-installation-of-kserve_serving-large-models)
	 - Red Hat Service Mesh Operator
	 - Red Hat Serverless Operator
	 - OpenShift AI 2.11 with KServe component enabled.
 * Set up local storage, so that you have a PersistentVolumeClaim with space for the llama-2-70b model files.
 
### FP8 quantization instructions
Before deploying the model, you need to convert the Llama-2-70b model to FP8 as described [in the vLLM documentation](https://docs.vllm.ai/en/latest/quantization/fp8.html#offline-quantization-with-static-activation-scaling-factors).  We use the `quantize-autofp8.py` Python script in this directory to do the conversion from the original model to FP8, which is based on the script in the docs. This can be done in an OpenShift Pod or on a separate system.

This script uses the AutoFP8 library, and also depends on having `torch` and 
`transformers` installed in your environment.
 
Update the paths to your model files in your script before running.
 ```
 ...
pretrained_model_dir = "Llama-2-70b-chat-hf"
quantized_model_dir = "Llama-2-70b-chat-hf-fp8"
...
```

After generating the FP8 model checkpoint, copy it to the PVC on your OpenShift cluster. The model deployment YAML assumes that the model files are in a PVC called `model-storage` which contains the model checkpoint in the directory `Llama-2-70b-chat-hf-fp8`.

## Deploying the model Pod
To deploy the model, first create the ServingRuntime custom resource using the YAML file in this directory:
```
oc apply -f servingruntime.yaml
```

Then create the InferenceService:
```
oc apply -f isvc.yaml
```
note:
- When running on 8xH100, we use 4 separate instances of the inference service, to create 4 copies of the model, by duplicating the isvc YAML and appending `-2`, `-3`, `-4` to the `metadata.name` field. When running on 4xL40S, we created two copies in the same way.
- If the model is stored in a different PVC, update the name in the InferenceService YAML  `spec.predictor.model.storageURI` field. Model can also be loaded from S3, see the [KServe documentation.](https://kserve.github.io/website/master/modelserving/storage/s3/s3/)


## Running the benchmarks

Deploy the test Pod on the cluster:
```
oc apply -f benchmark.yaml
```
and start a shell session inside the container:
```
oc exec -it pod/mlperf-inference -- /bin/bash
```

In the container, switch to the test directory:
```
cd /workspace/inference/language/llama2-70b/
```

We have updated some of the scripts from the reference implementation based on what we ran for our submission. See the `run_offline.sh, run_offline_accuracy.sh, run_server.sh, run_server_accuracy.sh` scripts. 

Before running the scripts, update the API_HOST and ADDITIONAL_SERVERS appropriately for your setup. 
Note: As a hack to split the batches of samples into  4 concurrent requests per server, we listed each server hostname 4 times. This allowed for some parallelization of the post-processing (converting returned text from vLLM back into token_ids), though this logic could be implemented in the code. For example:
```
API_HOST=<server-1>
ADDITIONAL_SERVERS='<server-1> <server-1> <server-1> <server-2> <server-2> <server-2> <server-2>'
```

For offline mode, also update the `--batch-size` argument appropriately for your system. 
- For 8xH100 GPU, we used `--batch-size=24576` , to send all of the samples in one shot
- For 4xL40S, we used `--batch-size=12288` to go through half of the samples  at a time

Once these variables are set corretly in the script, you can run the script to begin the test:
```
bash run_offline.sh
```
