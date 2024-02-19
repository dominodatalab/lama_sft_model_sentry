# LAMA2 SFT Model with integration with Model Sentry

This example adapts the LAMA2 SFT model notebook to train dnd deploy to Model Registry for integration with Model Sentry

The main challenge is finding shared storage between Workspace/Jobs in Domino and Model API. This is tackled by using
[Domsed](https://github.com/cerebrotech/domsed) to use the NFS mount used for Domino Datasets.

The mount is unique to each project in Domino and is mounted a follows in the Domino workloads and Model API

```shell
/artifacts/mlflow/{DOMINO_PROJECT_ID}
```

## Installation of Domsed

Follow these steps:

1. Clone the [repo](https://github.com/dominodatalab/domino-field-solutions-installations)
2. Run these steps in the repo

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