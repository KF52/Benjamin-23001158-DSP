from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, Spacer, Image
from io import BytesIO
from datetime import datetime

def generate_invoice_pdf(invoice):
    """Generate a professional-looking PDF invoice"""
    # Create a buffer and canvas
    buffer = BytesIO()
    pagesize = A4  # (595.27, 841.89) points
    c = canvas.Canvas(buffer, pagesize=pagesize)
    width, height = pagesize
    
    # Define colors
    primary_color = colors.HexColor('#2c3e50')  # Dark blue
    secondary_color = colors.HexColor('#3498db')  # Light blue
    text_color = colors.HexColor('#333333')  # Dark gray
    light_color = colors.HexColor('#ecf0f1')  # Light gray
    
    # Set up styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='InvoiceTitle',
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=primary_color,
        alignment=1,  # Center
        spaceAfter=0.2*inch
    ))
    styles.add(ParagraphStyle(
        name='InvoiceInfo',
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color
    ))
    styles.add(ParagraphStyle(
        name='InvoiceInfoBold',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=text_color
    ))
    styles.add(ParagraphStyle(
        name='TableHeader',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.white,
        alignment=1  # Center
    ))
    
    # Draw the header
    # Logo could be added here if available
    # c.drawImage("path/to/logo.png", 50, height - 100, width=100, height=50)
    
    # Draw header background
    c.setFillColor(primary_color)
    c.rect(0, height - 120, width, 120, fill=1, stroke=0)
    
    # Invoice title
    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 24)
    c.drawCentredString(width/2, height - 60, "INVOICE")
    
    # Invoice number
    c.setFont('Helvetica', 12)
    c.drawCentredString(width/2, height - 80, f"#{invoice.id}")
    
    # Organization info in header
    c.setFont('Helvetica', 10)
    c.drawString(50, height - 40, "InsurIQ - Machine Learning as a Service")
    c.drawString(50, height - 55, "UWE Bristol")
    c.drawString(50, height - 70, "Bristol, UK")
    
    # Status badge
    if invoice.status == 'Paid':
        status_color = colors.HexColor('#27ae60')  # Green
        status_text = "PAID"
    else:
        status_color = colors.HexColor('#e74c3c')  # Red
        status_text = "PENDING"
    
    # Draw status badge
    c.setFillColor(status_color)
    badge_width = 100
    badge_height = 30
    badge_x = width - badge_width - 50
    badge_y = height - 50
    c.roundRect(badge_x, badge_y, badge_width, badge_height, 5, fill=1, stroke=0)
    
    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 14)
    c.drawCentredString(badge_x + badge_width/2, badge_y + 10, status_text)
    
    # Invoice information section
    y_position = height - 150
    
    # Left column - Bill to
    c.setFillColor(text_color)
    c.setFont('Helvetica-Bold', 12)
    c.drawString(50, y_position, "BILL TO")
    
    c.setFont('Helvetica', 10)
    c.drawString(50, y_position - 20, invoice.user.name if hasattr(invoice.user, 'name') and invoice.user.name else invoice.user.email)
    c.drawString(50, y_position - 35, invoice.user.email)
    
    # Right column - Invoice details
    col_width = 175
    detail_x = width - col_width - 50
    
    c.setFont('Helvetica-Bold', 10)
    c.drawString(detail_x, y_position, "INVOICE DATE")
    c.drawString(detail_x, y_position - 30, "DUE DATE")
    c.drawString(detail_x, y_position - 60, "AMOUNT DUE")
    
    c.setFont('Helvetica', 10)
    # Format dates
    issued_date = invoice.issued_date.strftime("%d %b %Y") if hasattr(invoice, 'issued_date') else "N/A"
    due_date = invoice.due_date.strftime("%d %b %Y") if hasattr(invoice, 'due_date') else "N/A"
    
    c.drawString(detail_x + 100, y_position, issued_date)
    c.drawString(detail_x + 100, y_position - 30, due_date)
    
    # Amount with larger font and bold
    c.setFont('Helvetica-Bold', 14)
    c.drawString(detail_x + 100, y_position - 60, f"£{invoice.amount:.2f}")
    
    # Horizontal separator
    y_position -= 100
    c.setStrokeColor(light_color)
    c.setLineWidth(1)
    c.line(50, y_position, width - 50, y_position)
    
    # Invoice items table
    y_position -= 30
    c.setFont('Helvetica-Bold', 12)
    c.drawString(50, y_position, "INVOICE ITEMS")
    
    # Create table for invoice items
    y_position -= 20
    data = [
        ["DESCRIPTION", "AMOUNT"],
        [invoice.description, f"£{invoice.amount:.2f}"],
    ]
    
    # Set table style
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), secondary_color),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('ALIGN', (1, 1), (1, 1), 'RIGHT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (1, 0), 8),
        ('TOPPADDING', (0, 0), (1, 0), 8),
        ('GRID', (0, 0), (1, 1), 0.5, light_color),
    ])
    
    # Create and draw table
    table_width = width - 100
    table = Table(data, colWidths=[table_width * 0.7, table_width * 0.3])
    table.setStyle(table_style)
    
    # Draw table
    table.wrapOn(c, width, height)
    table.drawOn(c, 50, y_position - 30)
    
    # Total
    y_position -= 60
    c.setFont('Helvetica-Bold', 12)
    c.drawString(width - 150, y_position, "TOTAL:")
    c.drawString(width - 80, y_position, f"£{invoice.amount:.2f}")
    
    # Footer
    y_position = 50
    c.setFillColor(light_color)
    c.rect(0, 0, width, y_position + 20, fill=1, stroke=0)
    
    c.setFillColor(text_color)
    c.setFont('Helvetica', 9)
    c.drawCentredString(width/2, y_position, "Thank you for using InsurIQ - Machine Learning as a Service")
    c.drawCentredString(width/2, y_position - 15, "For any questions, please contact support@insuriq.com")
    
    # Payment information
    if invoice.status == 'Pending':
        y_position = 100
        c.setFillColor(colors.HexColor('#f39c12'))  # Orange
        c.roundRect(50, y_position, width - 100, 40, 5, fill=1, stroke=0)
        
        c.setFillColor(colors.white)
        c.setFont('Helvetica-Bold', 10)
        c.drawCentredString(width/2, y_position + 25, "PAYMENT INFORMATION")
        c.setFont('Helvetica', 9)
        c.drawCentredString(width/2, y_position + 10, "Please pay this invoice by the due date to continue using our services.")
    
    # Finalize the PDF
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def generate_prediction_pdf(prediction):
    """Generate a professional-looking PDF report for a prediction"""
    # Create a buffer and canvas
    buffer = BytesIO()
    pagesize = A4  # (595.27, 841.89) points
    c = canvas.Canvas(buffer, pagesize=pagesize)
    width, height = pagesize
    
    # Define colors
    primary_color = colors.HexColor('#2c3e50')  # Dark blue
    secondary_color = colors.HexColor('#3498db')  # Light blue
    text_color = colors.HexColor('#333333')  # Dark gray
    light_color = colors.HexColor('#ecf0f1')  # Light gray
    highlight_color = colors.HexColor('#27ae60')  # Green for positive values
    
    # Set up styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ReportTitle',
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=primary_color,
        alignment=1,  # Center
        spaceAfter=0.2*inch
    ))
    
    # Draw header background
    c.setFillColor(primary_color)
    c.rect(0, height - 120, width, 120, fill=1, stroke=0)
    
    # Report title
    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 24)
    c.drawCentredString(width/2, height - 60, "SETTLEMENT PREDICTION REPORT")
    
    # Prediction ID
    c.setFont('Helvetica', 12)
    c.drawCentredString(width/2, height - 80, f"Report #{prediction.id}")
    
    # Organization info in header
    c.setFont('Helvetica', 10)
    c.drawString(50, height - 40, "InsurIQ - Machine Learning as a Service")
    c.drawString(50, height - 55, "UWE Bristol")
    c.drawString(50, height - 70, "Bristol, UK")
    
    # Date on the right
    timestamp = prediction.timestamp.strftime("%d %b %Y, %H:%M") if hasattr(prediction, 'timestamp') else datetime.now().strftime("%d %b %Y, %H:%M")
    c.drawRightString(width - 50, height - 40, f"Date: {timestamp}")
    
    # User info
    y_position = height - 150
    c.setFillColor(text_color)
    c.setFont('Helvetica-Bold', 12)
    c.drawString(50, y_position, "CLIENT INFORMATION")
    
    c.setFont('Helvetica', 10)
    c.drawString(50, y_position - 20, f"Name: {prediction.user.name if hasattr(prediction.user, 'name') else 'N/A'}")
    c.drawString(50, y_position - 35, f"Email: {prediction.user.email if hasattr(prediction.user, 'email') else 'N/A'}")
    
    # Horizontal separator
    y_position -= 50
    c.setStrokeColor(light_color)
    c.setLineWidth(1)
    c.line(50, y_position, width - 50, y_position)
    
    # Prediction Result Section
    y_position -= 30
    c.setFillColor(primary_color)
    c.setFont('Helvetica-Bold', 14)
    c.drawString(50, y_position, "PREDICTION RESULT")
    
    # Settlement Value - Highlighted
    y_position -= 30
    c.setFillColor(highlight_color)
    c.roundRect(50, y_position - 10, 200, 40, 5, fill=1, stroke=0)
    
    c.setFillColor(colors.white)
    c.setFont('Helvetica-Bold', 12)
    c.drawString(60, y_position + 15, "SETTLEMENT VALUE")
    c.setFont('Helvetica-Bold', 18)
    c.drawString(60, y_position - 5, f"£{prediction.settlement_value:.2f}")
    
    # Input Data Section
    y_position -= 60
    c.setFillColor(primary_color)
    c.setFont('Helvetica-Bold', 14)
    c.drawString(50, y_position, "CLAIM DETAILS")
    
    # Create a table with input data
    y_position -= 30
    
    # Extract input data fields
    input_data = prediction.input_data
    table_data = [["FACTOR", "VALUE"]]
    
    # Add all input fields from the prediction
    for key, value in input_data.items():
        # Format the key with spaces before uppercase letters
        formatted_key = ' '.join([word.capitalize() for word in key.replace('_', ' ').split()])
        table_data.append([formatted_key, str(value)])
    
    # Set table style
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), secondary_color),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (1, 0), 8),
        ('TOPPADDING', (0, 0), (1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, light_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    
    # Calculate table height and adjust if it exceeds the page
    table_width = width - 100
    table = Table(table_data, colWidths=[table_width * 0.5, table_width * 0.5])
    table.setStyle(table_style)
    
    # Draw table
    table.wrapOn(c, width, height)
    table.drawOn(c, 50, y_position - (len(table_data) * 20))
    
    # Adjust y_position after the table
    y_position -= (len(table_data) * 20 + 40)
    
    # If we have feedback data, display it
    if hasattr(prediction, 'is_reasonable') and prediction.is_reasonable is not None:
        # Feedback Section
        c.setFillColor(primary_color)
        c.setFont('Helvetica-Bold', 14)
        c.drawString(50, y_position, "FEEDBACK INFORMATION")
        
        y_position -= 30
        c.setFont('Helvetica-Bold', 10)
        c.setFillColor(text_color)
        
        # Is reasonable
        reasonable_text = "Yes" if prediction.is_reasonable else "No"
        c.drawString(50, y_position, f"Assessment reasonable: {reasonable_text}")
        
        # Proposed settlement if available
        if hasattr(prediction, 'proposed_settlement') and prediction.proposed_settlement:
            y_position -= 20
            c.drawString(50, y_position, f"Proposed settlement: £{prediction.proposed_settlement:.2f}")
        
        # Adjustment rationale if available
        if hasattr(prediction, 'adjustment_rationale') and prediction.adjustment_rationale:
            y_position -= 20
            c.setFont('Helvetica-Bold', 10)
            c.drawString(50, y_position, "Adjustment rationale:")
            
            # Wrap long text
            y_position -= 15
            c.setFont('Helvetica', 9)
            text = prediction.adjustment_rationale
            wrapped_text = []
            max_width = width - 100
            current_line = ""
            for word in text.split():
                test_line = current_line + " " + word if current_line else word
                if c.stringWidth(test_line, 'Helvetica', 9) <= max_width:
                    current_line = test_line
                else:
                    wrapped_text.append(current_line)
                    current_line = word
            if current_line:
                wrapped_text.append(current_line)
            
            for line in wrapped_text:
                c.drawString(50, y_position, line)
                y_position -= 12
    
    # Footer
    footer_y = 50
    c.setFillColor(light_color)
    c.rect(0, 0, width, footer_y + 20, fill=1, stroke=0)
    
    c.setFillColor(text_color)
    c.setFont('Helvetica', 9)
    c.drawCentredString(width/2, footer_y, "This report was generated by InsurIQ - Machine Learning as a Service")
    c.drawCentredString(width/2, footer_y - 15, "For questions about this prediction, please contact support@insuriq.com")
    
    # Disclaimer
    disclaimer_y = 100
    disclaimer_text = "DISCLAIMER: This prediction is based on machine learning models and historical data. The actual settlement amount may vary based on additional factors not captured in this model."
    
    # Draw disclaimer in a box
    c.setFillColor(colors.HexColor('#f8f9fa'))
    c.roundRect(50, disclaimer_y - 40, width - 100, 50, 5, fill=1, stroke=0)
    
    c.setFillColor(text_color)
    c.setFont('Helvetica-Oblique', 8)
    
    # Wrap disclaimer text
    wrapped_disclaimer = []
    current_line = ""
    max_width = width - 120
    for word in disclaimer_text.split():
        test_line = current_line + " " + word if current_line else word
        if c.stringWidth(test_line, 'Helvetica-Oblique', 8) <= max_width:
            current_line = test_line
        else:
            wrapped_disclaimer.append(current_line)
            current_line = word
    if current_line:
        wrapped_disclaimer.append(current_line)
    
    disclaimer_line_y = disclaimer_y
    for line in wrapped_disclaimer:
        c.drawCentredString(width/2, disclaimer_line_y, line)
        disclaimer_line_y -= 12
    
    # Finalize the PDF
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer