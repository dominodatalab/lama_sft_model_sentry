# LAMA2 SFT Model with integration with Model Sentry

This example adapts the LAMA2 SFT model notebook to train dnd deploy to Model Registry for integration with Model Sentry

The main challenge is finding shared storage between Workspace/Jobs in Domino and Model API. This is tackled by using
[Domsed](https://github.com/cerebrotech/domsed) to use the NFS mount used for Domino Datasets.

The mount is unique to each project in Domino and is mounted a follows in the Domino workloads and Model API

```shell
/artifacts/mlflow/{DOMINO_PROJECT_ID}
```

The following notebooks are included-

1. Fine Tuning the LAMA SFT Model [notebook](https://github.com/dominodatalab/lama_sft_model_sentry/blob/main/llama2-ft.ipynb)
2. Clone the [repo](https://github.com/dominodatalab/domino-field-solutions-installations). The model checkpoints are written to `/artifacts/mlflow/{DOMINO_PROJECT_ID}`
3.  The pyfunc model is located in [my_model.py](https://github.com/dominodatalab/lama_sft_model_sentry/blob/main/my_model.py). This file reads the checkpoints from `/artifacts/mlflow/{DOMINO_PROJECT_ID}`
4.  Register the model to MLFLOW Model Registry, using the [notebook](https://github.com/dominodatalab/lama_sft_model_sentry/blob/main/register_model.ipynb) , `register_model.ipynb`.
5. Now you can use Model Sentry to view Model Cards, review model and deploy to Model API endpoint using the same Environment used to create the workspace (with the addition of libraries if any needed to run the model)


## Installation of Domsed

Follow these steps:

1. 

```shell
cd domsed
export platform_namespace=domino-platform
export compute_namespace=domino-compute
helm install -f values.yaml domsed helm/domsed -n ${platform_namespace}
kubectl label namespace ${compute_namespace} operator-enabled=true
```

## Install the mutation
```shell
kubectl -n domino-platform create -f mutation.yaml
```

For this basic demo, the mutation applies the "datasets" NFS mount into both workspaces and model api as follows:
`/artifacts/mlflow/{DOMINO_PROJECT_ID}` ensuring the workloads only see the sub-folder for their projects. The Domino workloads will not log LLM artifacts (such as checkpoints) to MLFLOW Artifact Store but instead write them directly to this location.

The following refinements are possible which will serve other Non Functional Requirements like managing the integrity of the artifacts-
1. The MODEL API will mount the NFS mount as a readonly folder
2. Instead of mounting a folder `/artifacts/mlflow/{DOMINO_PROJECT_ID}` we could mount the folder `/artifacts/mlflow/{DOMINO_PROJECT_ID}/{DOMINO_RUN_ID}` into a Workspace but mount `/artifacts/mlflow/{DOMINO_PROJECT_ID}/` into the Model API. This ensures that the artifacts cannot be modified afer a workspace stops. MLFLOW runs track the `DOMINO_RUN_ID` which can be used by the Model API to discover the artifacts loaded by the MLFLOW run from inside the workspace.

The takeaway being, the approach is flexible and can adapt to complex requirements while still ensuring large LLM artifacts are not copied.


   


## Environment

Base Env - `nvcr.io/nvidia/pytorch:22.12-py3`

Dockerfile

```
# System-level dependency injection runs as root
USER root:root

# Validate base image pre-requisites
# Complete requirements can be found at
# https://docs.dominodatalab.com/en/latest/user_guide/a00d1b/automatic-adaptation-of-custom-images/#_pre_requisites_for_automatic_custom_image_compatibility_with_domino
RUN /opt/domino/bin/pre-check.sh

# Configure /opt/domino to prepare for Domino executions
RUN /opt/domino/bin/init.sh

# Validate the environment
RUN /opt/domino/bin/validate.sh


RUN pip install -q -U trl==0.7.1 transformers==4.33.2 accelerate==0.23.0 peft==0.6.0
RUN pip install -q datasets==2.14.5 bitsandbytes==0.41.1 einops==0.6.1 mlflow==2.7.1 evaluate==0.4.0
RUN pip install --no-cache-dir Flask Flask-Compress Flask-Cors jsonify uWSGI 

RUN pip uninstall --yes transformer-engine
RUN pip uninstall -y apex

RUN pip uninstall --yes torch torchvision torchaudio

RUN pip install torch==2.0.1  --index-url https://download.pytorch.org/whl/cu118
```
