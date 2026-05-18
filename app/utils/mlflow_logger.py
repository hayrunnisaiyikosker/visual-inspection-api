import mlflow
import mlflow.sklearn
from typing import Optional

EXPERIMENT_NAME = "visual-inspection-api"

mlflow.set_experiment(EXPERIMENT_NAME)


def log_inference(
    endpoint: str,
    model_name: str,
    processing_time_ms: float,
    extra_metrics: Optional[dict] = None,
    extra_params: Optional[dict] = None,
):
    with mlflow.start_run(run_name=endpoint, nested=True):
        mlflow.set_tag("endpoint", endpoint)
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("processing_time_ms", processing_time_ms)

        if extra_metrics:
            for key, value in extra_metrics.items():
                mlflow.log_metric(key, value)

        if extra_params:
            for key, value in extra_params.items():
                mlflow.log_param(key, str(value))
