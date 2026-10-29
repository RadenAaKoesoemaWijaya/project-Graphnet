import base64
import re

import plotly.graph_objects as go
import plotly.io as plotly_io

from pdf_generator import PDFReportGenerator


_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/"
    "lZ8AAAAASUVORK5CYII="
)


def test_plotly_figure_exports_pdf_ready_png():
    figure = go.Figure(go.Bar(x=["train", "test"], y=[8, 2]))

    restored_figure = plotly_io.from_json(figure.to_json())
    image = restored_figure.to_image(format="png", width=1200, height=800, scale=2)

    assert image.startswith(b"\x89PNG\r\n\x1a\n")


def test_pdf_includes_unmapped_charts_and_tables():
    generator = PDFReportGenerator()
    report_data = {
        "metadata": {"generated_at": "2026-10-29", "language": "Indonesia"},
        "executive_summary": "Ringkasan evaluasi.",
        "preprocessing_analysis": "Preprocessing dilakukan pada partisi train.",
        "dataset_analysis": "Dataset evaluasi.",
        "feature_analysis": "Fitur terpilih.",
        "model_architecture": "Model ensemble.",
        "training_process": "Training selesai.",
        "performance_metrics": "Metrik tersedia.",
        "evaluation_metrics": {"Accuracy": 0.95},
        "recommendations": "Lanjutkan monitoring.",
    }
    chart_assets = {"preprocessing_distribution": _PIXEL_PNG}
    table_assets = {
        "preprocessing_column_profile": [
            ["Kolom", "Tipe", "Missing"],
            ["amount", "float", "x" * 12000],
        ],
        "evaluation_long_table": [
            ["Fitur", "Nilai"],
            *[[f"feature_{index}", str(index)] for index in range(100)],
        ],
    }
    report_data["charts"] = chart_assets
    report_data["tables"] = table_assets

    pdf = generator.generate_pdf(report_data).getvalue()

    assert pdf.startswith(b"%PDF")
    assert b"/Subtype /Image" in pdf
    assert len(re.findall(rb"/Type\s*/Page\b", pdf)) > 1
