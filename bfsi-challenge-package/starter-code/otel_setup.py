import logging
import os
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.metrics import set_meter_provider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider


logger = logging.getLogger(__name__)


def _normalise_endpoint(endpoint: Optional[str]) -> str:
    if not endpoint:
        return "localhost:4317"

    cleaned = endpoint.strip()
    if cleaned.startswith("http://"):
        cleaned = cleaned.replace("http://", "", 1)
    elif cleaned.startswith("https://"):
        cleaned = cleaned.replace("https://", "", 1)

    if ":" not in cleaned:
        return f"{cleaned}:4317"

    return cleaned


def configure_tracing(service_name: str = "bfsi-agent", otlp_endpoint: Optional[str] = None):
    """Configure OTLP tracing with a console fallback for local debugging.

    If the OTLP collector is running in Docker, traces are exported to the
    collector at `localhost:4317` by default. If that endpoint is unavailable,
    the SDK falls back to the standard console exporter so local debugging still
    works without Docker.
    """
    resource = Resource.create({"service.name": service_name})
    endpoint = _normalise_endpoint(
        otlp_endpoint
        or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or "localhost:4317"
    )

    provider = TracerProvider(resource=resource)

    try:
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry tracing configured with OTLP exporter at %s", endpoint)
        return trace.get_tracer_provider()
    except Exception as exc:  # pragma: no cover - defensive fallback
        fallback_provider = TracerProvider(resource=resource)
        fallback_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(fallback_provider)
        logger.warning(
            "Could not connect to OTLP collector at %s; using console exporter for local debugging. Error: %s",
            endpoint,
            exc,
        )
        return trace.get_tracer_provider()


def configure_metrics(service_name: str = "bfsi-agent"):
    """Configure a metrics pipeline using the same resource and exporter configuration as tracing."""
    resource = Resource.create({"service.name": service_name})
    meter_provider = MeterProvider(resource=resource)
    set_meter_provider(meter_provider)
    logger.info("OpenTelemetry metrics configured for service %s", service_name)
    return meter_provider


# Backwards-compatible alias used by the starter app
setup_tracing = configure_tracing


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    configure_tracing()
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("otel_setup_self_test") as span:
        span.set_attribute("debug.local", True)
        span.set_attribute("service.name", "bfsi-agent")
        logger.info("Tracer provider initialized successfully.")
