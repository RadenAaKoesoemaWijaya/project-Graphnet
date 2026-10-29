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


def test_wide_table_is_split_into_readable_horizontal_segments():
    generator = PDFReportGenerator()
    headers = ["identifier"] + [f"feature_{index}" for index in range(1, 75)]
    wide_table = [
        headers,
        *[
            [f"row_{row_index}"] + [
                f"value_{row_index}_{column_index}"
                for column_index in range(1, 75)
            ]
            for row_index in range(8)
        ],
    ]

    flowables = generator._add_table_section("Wide table", wide_table)
    tables = [flowable for flowable in flowables if hasattr(flowable, "_cellvalues")]
    rendered_headers = {
        cell.getPlainText()
        for table in tables
        for cell in table._cellvalues[0]
    }

    assert len(tables) == 15
    assert all(len(table._cellvalues[0]) <= 6 for table in tables)
    assert all(min(table._colWidths) > 4 for table in tables)
    assert set(headers).issubset(rendered_headers)

    report_data = {
        "metadata": {},
        "executive_summary": "Summary",
        "dataset_analysis": "Dataset",
        "feature_analysis": "Features",
        "model_architecture": "Model",
        "training_process": "Training",
        "performance_metrics": "Metrics",
        "recommendations": "Recommendations",
        "tables": {"wide_feature_summary": wide_table},
    }
    pdf = generator.generate_pdf(report_data).getvalue()

    assert pdf.startswith(b"%PDF")


def test_summary_pdf_contains_only_key_metrics_with_plain_language_meanings():
    generator = PDFReportGenerator()
    report_data = {
        "metadata": {"generated_at": "2026-10-30"},
        "executive_summary": "Model menandai sejumlah klaim untuk ditinjau.",
        "evaluation_metrics": {
            "Accuracy": 0.96,
            "Precision": 0.72,
            "Recall": 0.81,
            "F1-Score": 0.76,
            "F2-Score": 0.79,
            "PR-AUC": 0.68,
            "Matthews Correlation Coefficient": 0.4,
        },
    }
    tables = {
        "evaluation_prediction_summary": [
            ["Ringkasan", "Nilai"],
            ["Jumlah sampel evaluasi", "500"],
            ["Jumlah prediksi anomali", "40"],
            ["Tingkat anomali", "8.00%"],
            ["Threshold evaluasi", "0.50"],
        ],
        "evaluation_classification_report": [
            ["Kelas", "Precision", "Recall"],
            ["0", "0.96", "0.98"],
            ["1", "0.72", "0.81"],
        ],
        "evaluation_confusion_matrix": [
            ["Aktual / Prediksi", "Prediksi 0", "Prediksi 1"],
            ["Aktual 0", "450", "10"],
            ["Aktual 1", "8", "32"],
        ],
        "model_metrics": [
            ["Model", "Status", "Bobot", "Detail"],
            ["Isolation Forest", "Trained", "0.600", "unimportant internals"],
        ],
        "unrelated_large_table": [
            ["Raw Detail"],
            ["must not appear in concise output"],
        ],
    }

    story = generator._build_summary_story(report_data, charts={}, tables=tables)
    paragraph_text = "\n".join(
        flowable.getPlainText()
        for flowable in story
        if hasattr(flowable, "getPlainText")
    )
    table_text = "\n".join(
        str(cell.getPlainText() if hasattr(cell, "getPlainText") else cell)
        for flowable in story
        if hasattr(flowable, "_cellvalues")
        for row in flowable._cellvalues
        for cell in row
    ).replace("<br/>", " ")

    assert "Dari semua fraud aktual" in table_text
    assert "tertangkap" in table_text
    assert "false negative berarti fraud aktual terlewat." in paragraph_text.lower()
    assert "Akurasi" in table_text
    assert "Matthews Correlation Coefficient" not in table_text
    assert "must not appear in concise output" not in table_text
    assert "unimportant internals" not in table_text
    assert not any(flowable.__class__.__name__ == "PageBreak" for flowable in story)

    pdf = generator.generate_pdf(
        report_data,
        charts={},
        tables=tables,
        summary_only=True,
    ).getvalue()
    assert pdf.startswith(b"%PDF")


def test_summary_pdf_explains_when_supervised_metrics_are_unavailable():
    generator = PDFReportGenerator()
    story = generator._build_summary_story(
        {"metadata": {}, "evaluation_metrics": {}},
        charts={},
        tables={},
    )

    paragraph_text = "\n".join(
        flowable.getPlainText()
        for flowable in story
        if hasattr(flowable, "getPlainText")
    )
    assert "data evaluasi tidak memiliki label aktual" in paragraph_text
    assert "bukan ukuran akurasi model" in paragraph_text
