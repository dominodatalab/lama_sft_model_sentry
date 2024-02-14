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

