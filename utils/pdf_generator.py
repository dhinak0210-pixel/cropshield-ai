"""
CropShield AI — Luxury PDF Report Generator
==========================================
Generates high-fidelity, corporate-grade diagnostic PDF reports for crop pathology.
Uses dynamic metadata blocks, elegant forest green color schemes, and clean typographic grids.
"""

import datetime
from fpdf import FPDF

class CropShieldReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(15, 20, 15)
        self.set_auto_page_break(auto=True, margin=20)
        
    def header(self):
        # We only want running header on pages after the first page
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "CropShield Pathology Diagnostic Report", align="L")
            self.set_x(-40)
            self.cell(0, 10, f"Page {self.page_no()}", align="R")
            self.ln(8)
            # Draw a subtle separator line
            self.set_draw_color(220, 220, 220)
            self.line(15, self.get_y(), 195, self.get_y())
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        # Draw line above footer
        self.set_draw_color(220, 220, 220)
        self.line(15, self.get_y() - 2, 195, self.get_y() - 2)
        
        self.cell(0, 10, "CONFIDENTIAL - PathogenIQ Neural Precision Engine", align="L")
        self.set_x(-30)
        self.cell(0, 10, f"Page {self.page_no()}", align="R")

def sanitize_for_pdf(text: str) -> str:
    """
    Cleans and standardizes characters not supported by Helvetica (Latin-1 / CP1252)
    to prevent PDF generation errors.
    """
    if not text:
        return ""
        
    replacements = {
        "═": "=",
        "║": "|",
        "™": "TM",
        "©": "(c)",
        "®": "(R)",
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "•": "*",
        "·": "*",
        "…": "...",
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)
        
    # Also encode to latin1 and decode back, replacing other unsupported characters
    return text.encode("latin1", "replace").decode("latin1")

def generate_pdf_report(
    report_text: str,
    plant_name: str,
    disease_name: str,
    confidence: float,
    severity: str
) -> bytes:
    """
    Generates a beautifully typeset PDF report in memory and returns it as bytes.
    """
    pdf = CropShieldReportPDF()
    pdf.add_page()
    
    # 1. Luxury Header Banner (Top Block)
    pdf.set_fill_color(27, 67, 50)  # #1B4332 - Forest Green
    pdf.rect(0, 0, 210, 35, style="F")
    
    # Header Text
    pdf.set_y(10)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, "C R O P S H I E L D   A I", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(144, 219, 180)  # Light Mint green accent
    pdf.cell(0, 5, "PATHOLOGY DIAGNOSTIC REPORT | PATHOGENIQ PRECISION ENGINE", new_x="LMARGIN", new_y="NEXT", align="C")
    
    # 2. Spacing below banner
    pdf.set_y(42)
    
    # 3. Metadata Card (Light Green box with forest green border)
    pdf.set_fill_color(240, 247, 244)  # Warm white/green tint
    pdf.set_draw_color(64, 145, 108)   # Accent green
    pdf.set_line_width(0.5)
    pdf.rect(15, 42, 180, 42, style="FD")
    
    # Title of metadata card
    pdf.set_xy(20, 45)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(27, 67, 50)
    pdf.cell(0, 5, "DIAGNOSTIC ASSESSMENT METADATA", new_x="LMARGIN", new_y="NEXT")
    
    # Grid details
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    details = [
        [("Plant/Crop Type:", plant_name), ("Confidence Score:", f"{confidence:.1f}%")],
        [("Detected Pathogen:", disease_name), ("Estimated Severity:", severity.capitalize())],
        [("Analysis Timestamp:", current_time), ("Diagnostic Model:", "PathogenIQ Neural Engine v3.1")]
    ]
    
    y_start = 52
    for row in details:
        # Col 1
        pdf.set_xy(20, y_start)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(27, 67, 50)
        pdf.cell(32, 5, row[0][0])
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(45, 49, 46)
        pdf.cell(60, 5, sanitize_for_pdf(row[0][1]))
        
        # Col 2
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(27, 67, 50)
        pdf.cell(32, 5, row[1][0])
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(45, 49, 46)
        pdf.cell(50, 5, sanitize_for_pdf(row[1][1]))
        
        y_start += 7
        
    # 4. Spacing below metadata card
    pdf.set_y(90)
    
    # 5. Parse and draw report text
    sanitized_text = sanitize_for_pdf(report_text)
    lines = sanitized_text.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue
            
        # Skip original text report title/header block since we have our custom banner
        if (
            "PLANT DISEASE ASSESSMENT REPORT" in line
            or "Plant    :" in line
            or "Disease  :" in line
            or "Confidence:" in line
            or "Severity :" in line
            or "===" in line
        ):
            continue
            
        # Detect section headings like "1. EXECUTIVE SUMMARY" or "TREATMENT PLAN"
        # Often they start with digits, e.g. "1. " or "2. "
        is_heading = False
        if any(line.startswith(f"{i}. ") for i in range(1, 15)):
            is_heading = True
        
        if is_heading:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(27, 67, 50)
            
            # Left vertical indicator bar for luxury styling
            curr_y = pdf.get_y()
            pdf.set_fill_color(64, 145, 108)  # Secondary green
            pdf.rect(15, curr_y + 1, 3, 5, style="F")
            
            # Text indented slightly to the right of the bar
            pdf.set_x(20)
            pdf.cell(0, 7, line.upper(), new_x="LMARGIN", new_y="NEXT")
            
            # Add a bottom line under section title
            pdf.set_draw_color(220, 220, 220)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(3)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(45, 49, 46)
            
            # Check if line is bullet list item (starts with - or *)
            if line.startswith("-") or line.startswith("*"):
                # Indent bullets
                pdf.set_x(20)
                bullet_char = chr(149)  # Standard bullet character in cp1252
                pdf.cell(5, 5.5, bullet_char)
                text_to_print = line[1:].strip()
                pdf.multi_cell(0, 5.5, text_to_print)
            else:
                pdf.set_x(15)
                pdf.multi_cell(0, 5.5, line)
                
    # Return output bytes
    return bytes(pdf.output())
