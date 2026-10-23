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
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
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
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1E40AF'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        # Section heading style
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#3B82F6'),
            spaceAfter=12,
            spaceBefore=20
        ))

        # Subsection heading style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#64748B'),
            spaceAfter=8,
            spaceBefore=12
        ))

        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))

        # Metadata style
        self.styles.add(ParagraphStyle(
            name='Metadata',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#64748B'),
            alignment=TA_LEFT,
            spaceAfter=6
        ))

    def generate_pdf(
        self,
        report_data: Dict[str, Any],
        output_filename: str = None
    ) -> BytesIO:
        """
        Generate PDF report from training interpretation data.

        Args:
            report_data: Dictionary containing report sections from ModelInterpreter
            output_filename: Optional filename for the PDF

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
        story = self._build_story(report_data, story)

        doc.build(story)
        buffer.seek(0)
        return buffer

    def _build_story(self, report_data: Dict[str, Any], story: List) -> List:
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
        story.append(PageBreak())

        # Training process
        story.extend(self._build_section("Proses Pelatihan", report_data.get('training_process', '')))
        story.append(PageBreak())

        # Performance metrics
        story.extend(self._build_section("Metrik Performa", report_data.get('performance_metrics', '')))
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
            ('ALIGN', (0, 0), (-1, -1), TA_LEFT),
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
            self.styles['Metadata']
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
                    paragraphs.append(Paragraph(full_text, self.styles['BodyText']))
                    current_text = []
                continue

            # Handle headings
            if line.startswith('###'):
                if current_text:
                    full_text = ' '.join(current_text)
                    paragraphs.append(Paragraph(full_text, self.styles['BodyText']))
                    current_text = []
                heading_text = line.replace('###', '').strip()
                paragraphs.append(Paragraph(heading_text, self.styles['SubsectionHeading']))
                continue

            if line.startswith('##'):
                if current_text:
                    full_text = ' '.join(current_text)
                    paragraphs.append(Paragraph(full_text, self.styles['BodyText']))
                    current_text = []
                heading_text = line.replace('##', '').strip()
                paragraphs.append(Paragraph(heading_text, self.styles['SectionHeading']))
                continue

            # Handle bullet points
            if line.startswith('-'):
                if current_text:
                    full_text = ' '.join(current_text)
                    paragraphs.append(Paragraph(full_text, self.styles['BodyText']))
                    current_text = []
                bullet_text = line.replace('-', '•', 1).strip()
                paragraphs.append(Paragraph(bullet_text, self.styles['BodyText']))
                continue

            # Handle bold text
            line = line.replace('**', '<b>', 1).replace('**', '</b>', 1)

            current_text.append(line)

        # Add remaining text
        if current_text:
            full_text = ' '.join(current_text)
            paragraphs.append(Paragraph(full_text, self.styles['BodyText']))

        return paragraphs

    def _build_footer(self, report_data: Dict[str, Any]) -> List:
        """Build footer with metadata."""
        story = []

        metadata = report_data.get('metadata', {})
        footer_text = (
            f"Dokumen ini dibuat oleh ASTINA pada {metadata.get('generated_at', 'N/A')}. "
            f"Generated using {metadata.get('interpreter_provider', 'heuristic')} interpreter."
        )

        footer = Paragraph(footer_text, self.styles['Metadata'])
        story.append(footer)

        return story

    def save_pdf_to_file(
        self,
        report_data: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Generate PDF and save to file.

        Args:
            report_data: Dictionary containing report sections
            output_path: Path where PDF should be saved

        Returns:
            Path to the saved PDF file
        """
        buffer = self.generate_pdf(report_data)

        with open(output_path, 'wb') as f:
            f.write(buffer.getvalue())

        logger.info(f"PDF report saved to: {output_path}")
        return output_path


def generate_training_report_pdf(
    report_data: Dict[str, Any],
    output_path: str = None
) -> BytesIO:
    """
    Convenience function to generate PDF from report data.

    Args:
        report_data: Dictionary containing report sections from ModelInterpreter
        output_path: Optional path to save PDF file

    Returns:
        BytesIO buffer containing the PDF (if output_path is None)
        or path to saved file (if output_path is provided)
    """
    generator = PDFReportGenerator()

    if output_path:
        return generator.save_pdf_to_file(report_data, output_path)
    else:
        return generator.generate_pdf(report_data)
