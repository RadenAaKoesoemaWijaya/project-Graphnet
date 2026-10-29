"""
PDF Report Generator for ASTINA Model Training Reports

Generates professional PDF reports from model training interpretation results.
"""

import os
import logging
import textwrap
from html import escape
from datetime import datetime
from typing import Dict, Any, List
from io import BytesIO

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Image
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    _has_reportlab = True
except ImportError:
    _has_reportlab = False
    logging.warning("reportlab not installed. PDF generation will be disabled.")

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generates professional PDF reports for model training analysis."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        if not _has_reportlab:
            return

        # Title style - more elegant with gradient color
        if 'CustomTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=22,
                textColor=colors.HexColor('#0F172A'),
                spaceAfter=20,
                spaceBefore=10,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))

        # Subtitle style
        if 'CustomSubtitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomSubtitle',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#475569'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica'
            ))

        # Section heading style - modern blue
        if 'SectionHeading' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeading',
                parent=self.styles['Heading2'],
                fontSize=15,
                textColor=colors.HexColor('#1E40AF'),
                spaceAfter=15,
                spaceBefore=25,
                fontName='Helvetica-Bold',
                borderPadding=5,
                borderWidth=1,
                borderColor=colors.HexColor('#1E40AF')
            ))

        # Subsection heading style
        if 'SubsectionHeading' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SubsectionHeading',
                parent=self.styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#334155'),
                spaceAfter=10,
                spaceBefore=15,
                fontName='Helvetica-Bold'
            ))

        # Body text style - improved readability
        if 'CustomBodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomBodyText',
                parent=self.styles['Normal'],
                fontSize=10,
                leading=15,
                alignment=TA_JUSTIFY,
                spaceAfter=10,
                fontName='Helvetica'
            ))

        # Metadata style
        if 'CustomMetadata' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomMetadata',
                parent=self.styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#64748B'),
                alignment=TA_LEFT,
                spaceAfter=6,
                fontName='Helvetica'
            ))

        # Highlight style for important information
        if 'HighlightText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='HighlightText',
                parent=self.styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#059669'),
                spaceAfter=8,
                fontName='Helvetica-Oblique'
            ))

        # Table header style
        if 'TableHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='TableHeader',
                parent=self.styles['Normal'],
                fontSize=9,
                textColor=colors.white,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))

        # Footer style
        if 'FooterText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='FooterText',
                parent=self.styles['Normal'],
                fontSize=8,
                textColor=colors.HexColor('#94A3B8'),
                alignment=TA_CENTER,
                fontName='Helvetica'
            ))

    def generate_pdf(
        self,
        report_data: Dict[str, Any],
        output_filename: str = None,
        charts: Dict[str, bytes] = None,
        tables: Dict[str, List] = None
    ) -> BytesIO:
        """
        Generate PDF report from training interpretation data.

        Args:
            report_data: Dictionary containing report sections from ModelInterpreter
            output_filename: Optional filename for the PDF
            charts: Dictionary mapping chart names to image bytes (PNG/JPEG)
            tables: Dictionary mapping table names to table data (list of lists)

        Returns:
            BytesIO buffer containing the PDF data
        """
        if not _has_reportlab:
            raise ImportError("reportlab is not installed. Cannot generate PDF.")

        if charts is None:
            charts = report_data.get('charts')
        if tables is None:
            tables = report_data.get('tables')

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        story = []
        story = self._build_story(report_data, story, charts, tables)

        doc.build(story)
        buffer.seek(0)
        return buffer

    def _build_story(self, report_data: Dict[str, Any], story: List, charts: Dict[str, bytes] = None, tables: Dict[str, List] = None) -> List:
        """Build the PDF story from report data."""
        # Title page
        story.extend(self._build_title_page(report_data))
        story.append(PageBreak())

        # Executive summary
        story.extend(self._build_section("Ringkasan Eksekutif", report_data.get('executive_summary', '')))
        story.append(PageBreak())

        # Dataset analysis
        if report_data.get('preprocessing_analysis'):
            story.extend(self._build_section(
                "Tahap Data Preprocessing", report_data['preprocessing_analysis']
            ))
            story.append(PageBreak())

        story.extend(self._build_section("Analisis Dataset", report_data.get('dataset_analysis', '')))
        story.append(PageBreak())

        # Feature analysis
        story.extend(self._build_section("Analisis Fitur", report_data.get('feature_analysis', '')))
        story.append(PageBreak())

        # Model architecture
        story.extend(self._build_section("Arsitektur Model", report_data.get('model_architecture', '')))

        # Add ensemble weights chart if available
        if charts and 'ensemble_weights' in charts:
            story.append(Spacer(1, 0.3 * inch))
            story.extend(self._add_chart_section("Distribusi Bobot Ensemble", charts['ensemble_weights']))
            story.append(PageBreak())
        else:
            story.append(PageBreak())

        # Training process
        story.extend(self._build_section("Proses Pelatihan", report_data.get('training_process', '')))

        # Add training charts if available
        if charts:
            if 'autoencoder_loss' in charts:
                story.append(Spacer(1, 0.3 * inch))
                story.extend(self._add_chart_section("Grafik Loss Autoencoder", charts['autoencoder_loss']))
            if 'cross_validation' in charts:
                story.append(Spacer(1, 0.3 * inch))
                story.extend(self._add_chart_section("Grafik Cross-Validation", charts['cross_validation']))

        story.append(PageBreak())

        # Performance metrics
        story.extend(self._build_section("Metrik Performa", report_data.get('performance_metrics', '')))

        # Add performance tables if available
        if tables:
            if 'model_metrics' in tables:
                story.append(Spacer(1, 0.3 * inch))
                story.extend(self._add_table_section("Tabel Metrik Model", tables['model_metrics']))
            if 'optimization_results' in tables:
                story.append(Spacer(1, 0.3 * inch))
                story.extend(self._add_table_section("Hasil Optimasi Optuna", tables['optimization_results']))

        # Add GNN visualization if available
        if charts and 'gnn_subgraph' in charts:
            story.append(Spacer(1, 0.3 * inch))
            story.extend(self._add_chart_section("Visualisasi Jaringan Anomali GNN", charts['gnn_subgraph']))

        story.append(PageBreak())

        # Evaluation section (confusion matrix, ROC, etc.)
        if charts or report_data.get('evaluation_metrics'):
            story.extend(self._add_evaluation_section(report_data, charts))
            story.append(PageBreak())

        # Explainability section (SHAP, LIME)
        if charts and any(key in charts for key in ['shap_summary', 'shap_feature_importance', 'lime_explanation']):
            story.extend(self._add_explainability_section(report_data, charts))
            story.append(PageBreak())

        # Recommendations
        story.extend(self._build_section("Rekomendasi", report_data.get('recommendations', '')))

        # Include artifacts outside the original report's predefined sections.
        story.extend(self._add_remaining_artifacts(charts, tables))

        # Footer
        story.append(Spacer(1, 0.5 * inch))
        story.extend(self._build_footer(report_data))

        return story

    def _build_title_page(self, report_data: Dict[str, Any]) -> List:
        """Build title page with metadata."""
        story = []

        # Add decorative line at top
        line_table = Table([['']], colWidths=[6 * inch], rowHeights=[0.05 * inch])
        line_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1E40AF')),
        ]))
        story.append(line_table)
        story.append(Spacer(1, 0.5 * inch))

        # Main title
        title = Paragraph("Laporan Analisis Pelatihan Model", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.2 * inch))

        # Subtitle
        subtitle = Paragraph("ASTINA - Analisis Sistem Transaksi Identifikasi Nilai Anomali", self.styles['CustomSubtitle'])
        story.append(subtitle)
        story.append(Spacer(1, 0.8 * inch))

        # Metadata table with improved styling
        metadata = report_data.get('metadata', {})
        metadata_data = [
            ['Tanggal Pembuatan', metadata.get('generated_at', 'N/A')],
            ['Bahasa', metadata.get('language', 'Indonesia')],
            ['Interpreter Provider', metadata.get('interpreter_provider', 'heuristic')],
        ]

        metadata_table = Table(metadata_data, colWidths=[2.5 * inch, 3 * inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F8FAFC')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 1.2 * inch))

        # Disclaimer with better styling
        disclaimer_box = Paragraph(
            "<b>DISCLAIMER:</b> Dokumen ini dibuat secara otomatis oleh sistem ASTINA. "
            "Informasi yang disajikan berdasarkan hasil pelatihan model dan analisis data yang tersedia. "
            "Selalu verifikasi dengan data aktual sebelum mengambil keputusan bisnis.",
            self.styles['CustomMetadata']
        )
        story.append(disclaimer_box)
        story.append(Spacer(1, 0.3 * inch))

        # Add decorative line at bottom
        story.append(line_table)

        return story

    def _build_section(self, title: str, content: str) -> List:
        """Build a section with title and content."""
        story = []

        # Section heading
        heading = Paragraph(title, self.styles['SectionHeading'])
        story.append(heading)
        story.append(Spacer(1, 0.2 * inch))

        # Content (convert markdown-like syntax to paragraphs)
        paragraphs = self._parse_markdown_to_paragraphs(content)
        for para in paragraphs:
            story.append(para)
            story.append(Spacer(1, 0.1 * inch))

        return story

    def _parse_markdown_to_paragraphs(self, text: str) -> List:
        """Parse markdown-like text to Paragraph objects."""
        paragraphs = []
        lines = str(text or '').split('\n')
        current_text = []

        for line in lines:
            line = line.strip()

            if not line:
                if current_text:
                    full_text = ' '.join(current_text)
                    paragraphs.append(Paragraph(full_text, self.styles['CustomBodyText']))
                    current_text = []
                continue

            # Handle headings
            if line.startswith('###'):
                if current_text:
                    full_text = ' '.join(current_text)
                    paragraphs.append(Paragraph(full_text, self.styles['CustomBodyText']))
                    current_text = []
                heading_text = line.replace('###', '').strip()
                paragraphs.append(Paragraph(heading_text, self.styles['SubsectionHeading']))
                continue

            if line.startswith('##'):
                if current_text:
                    full_text = ' '.join(current_text)
                    paragraphs.append(Paragraph(full_text, self.styles['CustomBodyText']))
                    current_text = []
                heading_text = line.replace('##', '').strip()
                paragraphs.append(Paragraph(heading_text, self.styles['SectionHeading']))
                continue

            # Handle bullet points
            if line.startswith('-'):
                if current_text:
                    full_text = ' '.join(current_text)
                    paragraphs.append(Paragraph(full_text, self.styles['CustomBodyText']))
                    current_text = []
                bullet_text = line.replace('-', '•', 1).strip()
                paragraphs.append(Paragraph(bullet_text, self.styles['CustomBodyText']))
                continue

            # Handle bold text
            line = line.replace('**', '<b>', 1).replace('**', '</b>', 1)

            current_text.append(line)

        # Add remaining text
        if current_text:
            full_text = ' '.join(current_text)
            paragraphs.append(Paragraph(full_text, self.styles['CustomBodyText']))

        return paragraphs

    def _build_footer(self, report_data: Dict[str, Any]) -> List:
        """Build footer with metadata."""
        story = []

        metadata = report_data.get('metadata', {})
        footer_text = (
            f"Dokumen ini dibuat oleh ASTINA pada {metadata.get('generated_at', 'N/A')}. "
            f"Generated using {metadata.get('interpreter_provider', 'heuristic')} interpreter."
        )

        footer = Paragraph(footer_text, self.styles['CustomMetadata'])
        story.append(footer)

        return story

    def _add_chart_section(self, title: str, chart_bytes: bytes) -> List:
        """Add a chart image to the PDF."""
        story = []

        # Section heading
        heading = Paragraph(title, self.styles['SectionHeading'])
        story.append(heading)
        story.append(Spacer(1, 0.2 * inch))

        # Add image from bytes
        try:
            from io import BytesIO
            img_buffer = BytesIO(chart_bytes)
            img = Image(img_buffer, width=6 * inch, height=4 * inch, kind='proportional')
            img.hAlign = 'CENTER'
            story.append(img)
        except Exception as e:
            logger.warning(f"Failed to add chart to PDF: {e}")
            story.append(Paragraph(f"[Grafik tidak dapat ditampilkan: {str(e)}]", self.styles['CustomBodyText']))

        return story

    def _add_table_section(self, title: str, table_data: List) -> List:
        """Add a data table to the PDF."""
        story = []

        # Section heading
        heading = Paragraph(title, self.styles['SectionHeading'])
        story.append(heading)
        story.append(Spacer(1, 0.2 * inch))

        # Create table
        try:
            if table_data and len(table_data) > 0:
                normalized_rows = [
                    list(row) if isinstance(row, (list, tuple)) else [row]
                    for row in table_data
                ]
                n_cols = max(len(row) for row in normalized_rows)
                normalized_rows = [
                    row + [''] * (n_cols - len(row)) for row in normalized_rows
                ]
                available_width = 6.27 * inch
                col_widths = (
                    [2.5 * inch] + [(available_width - 2.5 * inch) / (n_cols - 1)] * (n_cols - 1)
                    if n_cols > 1 else [available_width]
                )
                min_column_width = min(col_widths)
                cell_padding = min(8, max(2, (min_column_width - 4) / 2))
                cell_font_size = min(8, max(5, min_column_width * 0.45))
                header_font_size = min(9, max(5, min_column_width * 0.5))
                cell_style = ParagraphStyle(
                    'PDFTableCell',
                    parent=self.styles['Normal'],
                    fontSize=cell_font_size,
                    leading=cell_font_size + 2,
                    wordWrap='CJK',
                )
                header_style = ParagraphStyle(
                    'PDFTableHeader',
                    parent=cell_style,
                    textColor=colors.white,
                    alignment=TA_CENTER,
                    fontName='Helvetica-Bold',
                    fontSize=header_font_size,
                )
                formatted_rows = []
                for row_index, row in enumerate(normalized_rows):
                    style = header_style if row_index == 0 else cell_style
                    wrapped_cells = []
                    for value, width in zip(row, col_widths):
                        char_width = max(
                            3,
                            int(max(width - 2 * cell_padding, 3) / (cell_font_size * 0.55)),
                        )
                        wrapped_cells.append(textwrap.wrap(
                            str(value if value is not None else ''),
                            width=char_width,
                            break_long_words=True,
                            break_on_hyphens=True,
                        ) or [''])

                    if row_index == 0:
                        chunk_count = 1
                    else:
                        chunk_count = max(
                            (len(lines) + 39) // 40 for lines in wrapped_cells
                        )
                    for chunk_index in range(chunk_count):
                        cells = []
                        for column_index, lines in enumerate(wrapped_cells):
                            start = chunk_index * 40
                            chunk_lines = lines[start:start + 40]
                            if chunk_index and column_index == 0 and len(lines) <= start:
                                chunk_lines = lines[:40]
                            escaped_lines = [escape(line) for line in chunk_lines]
                            cells.append(Paragraph('<br/>'.join(escaped_lines), style))
                        formatted_rows.append(cells)

                table = Table(
                    formatted_rows,
                    colWidths=col_widths,
                    repeatRows=1,
                    splitByRow=1,
                )
                table.setStyle(TableStyle([
                    # Header row styling
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E40AF')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), header_font_size),
                    ('TOPPADDING', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('LEFTPADDING', (0, 0), (-1, 0), cell_padding),
                    ('RIGHTPADDING', (0, 0), (-1, 0), cell_padding),
                    # Data rows styling
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 1), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), cell_font_size),
                    ('TOPPADDING', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                    ('LEFTPADDING', (0, 1), (-1, -1), cell_padding),
                    ('RIGHTPADDING', (0, 1), (-1, -1), cell_padding),
                    # Grid styling
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
                    ('LINEBELOW', (0, 0), (-1, 0), 1.5, colors.HexColor('#1E40AF')),
                    # Alternating row colors
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
                ]))
                story.append(table)
            else:
                story.append(Paragraph("[Tabel kosong]", self.styles['CustomBodyText']))
        except Exception as e:
            logger.warning(f"Failed to add table to PDF: {e}")
            story.append(Paragraph(f"[Tabel tidak dapat ditampilkan: {str(e)}]", self.styles['CustomBodyText']))

        return story

    def _add_remaining_artifacts(
        self,
        charts: Dict[str, bytes] = None,
        tables: Dict[str, List] = None
    ) -> List:
        """Render every chart and table not already placed in a report section."""
        story = []
        displayed_charts = {
            'ensemble_weights', 'autoencoder_loss', 'cross_validation',
            'gnn_subgraph', 'confusion_matrix', 'roc_curve',
            'precision_recall_curve', 'shap_summary',
            'shap_feature_importance', 'lime_explanation',
        }
        displayed_tables = {'model_metrics', 'optimization_results'}
        remaining_charts = {
            name: image for name, image in (charts or {}).items()
            if name not in displayed_charts
        }
        remaining_tables = {
            name: data for name, data in (tables or {}).items()
            if name not in displayed_tables
        }
        if not remaining_charts and not remaining_tables:
            return story

        story.append(PageBreak())
        story.extend(self._build_section(
            "Lampiran Tabel dan Visualisasi",
            "Seluruh tabel dan visualisasi yang dihasilkan pada tahapan preprocessing, "
            "pelatihan, dan evaluasi yang belum ditampilkan pada bagian sebelumnya."
        ))
        for name, image in remaining_charts.items():
            if image:
                story.extend(self._add_chart_section(name.replace('_', ' ').title(), image))
                story.append(Spacer(1, 0.2 * inch))
        for name, table_data in remaining_tables.items():
            story.extend(self._add_table_section(name.replace('_', ' ').title(), table_data))
            story.append(Spacer(1, 0.2 * inch))
        return story

    def _add_evaluation_section(self, report_data: Dict[str, Any], charts: Dict[str, bytes] = None) -> List:
        """Add evaluation metrics section to the PDF."""
        story = []

        # Section heading
        heading = Paragraph("Evaluasi Model", self.styles['SectionHeading'])
        story.append(heading)
        story.append(Spacer(1, 0.2 * inch))

        # Add confusion matrix if available
        if charts and 'confusion_matrix' in charts:
            story.extend(self._add_chart_section("Confusion Matrix", charts['confusion_matrix']))
            story.append(Spacer(1, 0.3 * inch))

        # Add ROC curve if available
        if charts and 'roc_curve' in charts:
            story.extend(self._add_chart_section("ROC Curve", charts['roc_curve']))
            story.append(Spacer(1, 0.3 * inch))

        # Add precision-recall curve if available
        if charts and 'precision_recall_curve' in charts:
            story.extend(self._add_chart_section("Precision-Recall Curve", charts['precision_recall_curve']))
            story.append(Spacer(1, 0.3 * inch))

        # Add evaluation metrics table if available
        evaluation_metrics = report_data.get('evaluation_metrics', {})
        if evaluation_metrics:
            metrics_data = [['Metrik', 'Nilai']]
            for metric_name, metric_value in evaluation_metrics.items():
                if isinstance(metric_value, float):
                    metrics_data.append([metric_name, f"{metric_value:.4f}"])
                else:
                    metrics_data.append([metric_name, str(metric_value)])

            story.extend(self._add_table_section("Metrik Evaluasi", metrics_data))

        return story

    def _add_explainability_section(self, report_data: Dict[str, Any], charts: Dict[str, bytes] = None) -> List:
        """Add explainability section (SHAP, LIME) to the PDF."""
        story = []

        # Section heading
        heading = Paragraph("Explainability AI (SHAP & LIME)", self.styles['SectionHeading'])
        story.append(heading)
        story.append(Spacer(1, 0.2 * inch))

        # Add SHAP summary plot if available
        if charts and 'shap_summary' in charts:
            story.extend(self._add_chart_section("SHAP Summary Plot", charts['shap_summary']))
            story.append(Spacer(1, 0.3 * inch))

        # Add SHAP feature importance if available
        if charts and 'shap_feature_importance' in charts:
            story.extend(self._add_chart_section("SHAP Feature Importance", charts['shap_feature_importance']))
            story.append(Spacer(1, 0.3 * inch))

        # Add LIME explanation if available
        if charts and 'lime_explanation' in charts:
            story.extend(self._add_chart_section("LIME Explanation", charts['lime_explanation']))
            story.append(Spacer(1, 0.3 * inch))

        # Add explainability text description
        explainability_text = report_data.get('explainability_analysis', '')
        if explainability_text:
            paragraphs = self._parse_markdown_to_paragraphs(explainability_text)
            for para in paragraphs:
                story.append(para)
                story.append(Spacer(1, 0.1 * inch))

        return story

    def save_pdf_to_file(
        self,
        report_data: Dict[str, Any],
        output_path: str,
        charts: Dict[str, bytes] = None,
        tables: Dict[str, List] = None
    ) -> str:
        """
        Generate PDF and save to file.

        Args:
            report_data: Dictionary containing report sections
            output_path: Path where PDF should be saved
            charts: Dictionary mapping chart names to image bytes
            tables: Dictionary mapping table names to table data

        Returns:
            Path to the saved PDF file
        """
        buffer = self.generate_pdf(report_data, charts=charts, tables=tables)

        with open(output_path, 'wb') as f:
            f.write(buffer.getvalue())

        logger.info(f"PDF report saved to: {output_path}")
        return output_path


def generate_training_report_pdf(
    report_data: Dict[str, Any],
    output_path: str = None,
    charts: Dict[str, bytes] = None,
    tables: Dict[str, List] = None
) -> BytesIO:
    """
    Convenience function to generate PDF from report data.

    Args:
        report_data: Dictionary containing report sections from ModelInterpreter
        output_path: Optional path to save PDF file
        charts: Dictionary mapping chart names to image bytes (PNG/JPEG)
        tables: Dictionary mapping table names to table data (list of lists)

    Returns:
        BytesIO buffer containing the PDF (if output_path is None)
        or path to saved file (if output_path is provided)
    """
    generator = PDFReportGenerator()

    if output_path:
        return generator.save_pdf_to_file(report_data, output_path, charts=charts, tables=tables)
    else:
        return generator.generate_pdf(report_data, charts=charts, tables=tables)
