import os

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_PROVIDER: TracerProvider | None = None
_METRICS_PROVIDER: MeterProvider | None = None


def configure_otel() -> TracerProvider:
    global _PROVIDER

    if _PROVIDER is not None:
        return _PROVIDER

    resource = Resource.create({"service.name": "anvesh-agent"})
    provider = TracerProvider(resource=resource)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))

    try:
        trace.set_tracer_provider(provider)
    except Exception:  # pragma: no cover - provider already set by app bootstrap
        pass

    _PROVIDER = provider
    return provider


def configure_metrics() -> MeterProvider:
    global _METRICS_PROVIDER

    if _METRICS_PROVIDER is not None:
        return _METRICS_PROVIDER

    resource = Resource.create({"service.name": "anvesh-agent"})
    provider = MeterProvider(resource=resource, metric_readers=[
        PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318").rstrip('/') + "/v1/metrics"),
            export_interval_millis=5000,
        )
    ])

    try:
        metrics.set_meter_provider(provider)
    except Exception:  # pragma: no cover - provider already set by app bootstrap
        pass

    _METRICS_PROVIDER = provider
    return provider


def get_tracer(name: str):
    configure_otel()
    return trace.get_tracer(name)


def get_meter(name: str):
    configure_metrics()
    return metrics.get_meter(name)
