from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib import colors

c = canvas.Canvas('test_canvas.pdf', pagesize=(960, 540))
styles = getSampleStyleSheet()
style = ParagraphStyle('test', parent=styles['Normal'], fontSize=14, leading=18, textColor=colors.HexColor('#FFFFFF'))
p = Paragraph('<b>Hello</b> from <font color="#50E3C2">ReportLab</font> canvas!', style)
w, h = p.wrap(300, 100)
c.setFillColor(colors.HexColor('#0D1117'))
c.rect(0, 0, 960, 540, fill=1, stroke=0)
p.drawOn(c, 50, 480)
c.save()
print('Canvas test OK!')
