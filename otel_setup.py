from __future__ import annotations

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcSpanExporter
except ImportError:  # pragma: no cover
    OTLPGrpcSpanExporter = None

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpSpanExporter
except ImportError:  # pragma: no cover
    OTLPHttpSpanExporter = None


def _get_service_name() -> str:
    return os.getenv("OTEL_SERVICE_NAME", "treasury-liquidity-agent")


def configure_tracing(
    service_name: Optional[str] = None,
    collector_endpoint: Optional[str] = None,
    enable_console_fallback: bool = True,
) -> trace.Tracer:
    """
    Configure a TracerProvider using OTLP if the collector is reachable.
    If the collector endpoint is an HTTP URL (for example http://localhost:4318),
    use the HTTP exporter; otherwise use the gRPC exporter.
    """
    resource = Resource.create({"service.name": service_name or _get_service_name()})
    provider = TracerProvider(resource=resource)

    endpoint = collector_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    is_http_endpoint = endpoint.startswith("http://") or endpoint.startswith("https://")

    otlp_enabled = False
    if is_http_endpoint:
        if OTLPHttpSpanExporter is not None:
            try:
                http_endpoint = endpoint.rstrip("/")
                if "/v1/traces" not in http_endpoint:
                    http_endpoint = f"{http_endpoint}/v1/traces"
                exporter = OTLPHttpSpanExporter(endpoint=http_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                otlp_enabled = True
            except Exception:
                otlp_enabled = False
    else:
        if OTLPGrpcSpanExporter is not None:
            try:
                grpc_endpoint = endpoint.replace("http://", "", 1).replace("https://", "", 1)
                exporter = OTLPGrpcSpanExporter(endpoint=grpc_endpoint, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                otlp_enabled = True
            except Exception:
                otlp_enabled = False

    if enable_console_fallback and not otlp_enabled:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        print(
            f"[otel_setup] OTLP exporter unavailable at {endpoint}. "
            "Using console exporter for local debugging."
        )

    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name or _get_service_name())


if __name__ == "__main__":
    tracer = configure_tracing()
    with tracer.start_as_current_span("demo-span", kind=SpanKind.INTERNAL) as span:
        span.set_attribute("demo.attribute", "hello")
        span.set_status(Status(StatusCode.OK))
        print("Telemetry set up successfully.")
