"""
PDF Report Generator for ASTINA Model Training Reports

Generates professional PDF reports from model training interpretation results.
"""

import os
import logging
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

        # Title style
        if 'CustomTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=18,
                textColor=colors.HexColor('#1E40AF'),
                spaceAfter=30,
                alignment=TA_CENTER
            ))

        # Section heading style
        if 'SectionHeading' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeading',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#3B82F6'),
                spaceAfter=12,
                spaceBefore=20
            ))

        # Subsection heading style
        if 'SubsectionHeading' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SubsectionHeading',
                parent=self.styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#64748B'),
                spaceAfter=8,
                spaceBefore=12
            ))

        # Body text style (use unique name to avoid conflict)
        if 'CustomBodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomBodyText',
                parent=self.styles['Normal'],
                fontSize=10,
                leading=14,
                alignment=TA_JUSTIFY,
                spaceAfter=12
            ))

        # Metadata style
        if 'CustomMetadata' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomMetadata',
                parent=self.styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#64748B'),
                alignment=TA_LEFT,
                spaceAfter=6
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

        # Recommendations
        story.extend(self._build_section("Rekomendasi", report_data.get('recommendations', '')))

        # Footer
        story.append(Spacer(1, 0.5 * inch))
        story.extend(self._build_footer(report_data))

        return story

    def _build_title_page(self, report_data: Dict[str, Any]) -> List:
        """Build title page with metadata."""
        story = []

        # Main title
        title = Paragraph("Laporan Analisis Pelatihan Model", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.3 * inch))

        # Subtitle
        subtitle = Paragraph("ASTINA - Analisis Sistem Transaksi Identifikasi Nilai Anomali", self.styles['SectionHeading'])
        story.append(subtitle)
        story.append(Spacer(1, 0.5 * inch))

        # Metadata table
        metadata = report_data.get('metadata', {})
        metadata_data = [
            ['Tanggal Pembuatan', metadata.get('generated_at', 'N/A')],
            ['Bahasa', metadata.get('language', 'Indonesia')],
            ['Interpreter Provider', metadata.get('interpreter_provider', 'heuristic')],
        ]

        metadata_table = Table(metadata_data, colWidths=[2.5 * inch, 3 * inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F1F5F9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(metadata_table)
        story.append(Spacer(1, 1 * inch))

        # Disclaimer
        disclaimer = Paragraph(
            "<b>DISCLAIMER:</b> Dokumen ini dibuat secara otomatis oleh sistem ASTINA. "
            "Informasi yang disajikan berdasarkan hasil pelatihan model dan analisis data yang tersedia. "
            "Selalu verifikasi dengan data aktual sebelum mengambil keputusan bisnis.",
            self.styles['CustomMetadata']
        )
        story.append(disclaimer)

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
        lines = text.split('\n')
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
            img = Image(img_buffer, width=6 * inch, height=4 * inch)
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
                # Calculate column widths
                n_cols = len(table_data[0])
                col_widths = [2.5 * inch] + [(6 * inch - 2.5 * inch) / (n_cols - 1)] * (n_cols - 1) if n_cols > 1 else [6 * inch]

                table = Table(table_data, colWidths=col_widths)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E40AF')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')]),
                ]))
                story.append(table)
            else:
                story.append(Paragraph("[Tabel kosong]", self.styles['CustomBodyText']))
        except Exception as e:
            logger.warning(f"Failed to add table to PDF: {e}")
            story.append(Paragraph(f"[Tabel tidak dapat ditampilkan: {str(e)}]", self.styles['CustomBodyText']))

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
