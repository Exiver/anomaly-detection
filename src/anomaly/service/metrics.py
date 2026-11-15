from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from anomaly.service.drifts import DriftSnapshot


class ServiceMetrics:
    content_type = CONTENT_TYPE_LATEST

    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()

        self.requests = Counter(
            "ad_http_requests_total", "HTTP requests", ["method", "path", "status"], registry=r
        )
        self.request_seconds = Histogram(
            "ad_http_request_duration_seconds", "HTTP request latency", ["path"], registry=r
        )
        self.rejected = Counter(
            "ad_rejected_requests_total", "Requests rejected before inference", ["reason"], registry=r
        )
        self.in_flight = Gauge("ad_inferences_in_flight", "Inferences running now", registry=r)
        self.predictions = Counter(
            "ad_predictions_total", "Predictions by outcome", ["category", "outcome"], registry=r
        )
        self.inference_seconds = Histogram(
            "ad_inference_duration_seconds",
            "Model inference latency",
            ["category"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=r,
        )
        self.score_ratio = Histogram(
            "ad_score_ratio",
            "Anomaly score divided by the category threshold (>= 1 means anomalous)",
            ["category"],
            buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0),
            registry=r,
        )
        self.model_info = Gauge("ad_model_info", "Loaded release version", ["version"], registry=r)

        self.window_anomaly_rate = Gauge(
            "ad_window_anomaly_rate", "Anomaly rate over the recent window", ["category"], registry=r
        )
        self.window_mean_ratio = Gauge(
            "ad_window_mean_score_ratio", "Mean score ratio over the recent window", ["category"], registry=r
        )
        self.drift_ks_statistic = Gauge(
            "ad_drift_ks_statistic", "KS statistic: recent scores vs normal reference", ["category"], registry=r
        )
        self.drift_ks_pvalue = Gauge(
            "ad_drift_ks_pvalue", "Approximate KS p-value", ["category"], registry=r
        )
        self.input_mean_z = Gauge(
            "ad_input_mean_zscore", "Z-score of the recent mean input value vs training", ["category"], registry=r
        )

    def update_drift(self, category: str, snapshot: DriftSnapshot) -> None:
        self.window_anomaly_rate.labels(category).set(snapshot.anomaly_rate)
        self.window_mean_ratio.labels(category).set(snapshot.mean_score_ratio)
        if snapshot.ks_statistic is not None:
            self.drift_ks_statistic.labels(category).set(snapshot.ks_statistic)
            self.drift_ks_pvalue.labels(category).set(snapshot.ks_pvalue)
        if snapshot.input_mean_z is not None:
            self.input_mean_z.labels(category).set(snapshot.input_mean_z)

    def render(self) -> bytes:
        return generate_latest(self.registry)