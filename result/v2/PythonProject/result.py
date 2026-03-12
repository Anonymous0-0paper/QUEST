import os
import re
import time

import pymupdf  # PyMuPDF
import win32com.client

# Input/output paths
input_folder = r"C:\Users\Mehrab\OneDrive\Desktop\quest\new\results-excel"
output_root = r"C:\Users\Mehrab\OneDrive\Desktop\quest\new\results"

# Chart categories
categories = ["AoI", "Energy", "Makespan"]
pattern = re.compile(r"result-(.+)-(\d+)\.xlsx", re.IGNORECASE)


def crop_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    for page in doc:
        # Set the cropbox
        page.set_cropbox(pymupdf.Rect(4, 210, 612, 575))

    temp_path = pdf_path + ".temp.pdf"
    doc.save(temp_path, garbage=3, deflate=True)
    doc.close()
    os.remove(pdf_path)
    os.rename(temp_path, pdf_path)

files = os.listdir(input_folder)
j = 0
while j < len(files):
    file = files[j]
    if file.endswith(".xlsx") and file.startswith("result-"):
        match = pattern.match(file)
        if not match:
            print(f"Skipping malformed filename: {file}")
            continue

        # Create Excel instance
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False

        base_name, number = match.groups()
        wb_path = os.path.join(input_folder, file)
        wb = excel.Workbooks.Open(wb_path)
        pdfs = []

        try:
            sheet = wb.Sheets("Charts")
            chart_objs = sheet.ChartObjects()

            for i in range(min(3, chart_objs.Count)):  # Only 3 categories
                category = categories[i]
                chart_obj = chart_objs.Item(i + 1)  # 1-based indexing in COM
                print(base_name, number, category)

                # Create necessary folder path
                export_dir = os.path.join(output_root, category, number)
                os.makedirs(export_dir, exist_ok=True)

                export_path = os.path.join(export_dir, f"{base_name}.pdf")

                # Copy chart to temporary sheet
                temp_sheet = wb.Sheets.Add()
                chart_obj.Copy()
                temp_sheet.Paste()
                excel.CutCopyMode = False

                # Resize pasted chart (optional)
                pasted_chart = temp_sheet.ChartObjects(1)
                pasted_chart.Width = chart_obj.Width
                pasted_chart.Height = chart_obj.Height
                pasted_chart.Left = 100
                pasted_chart.Top = 50

                # Export to PDF
                temp_sheet.ExportAsFixedFormat(0, export_path)

                # Delete the temp sheet
                time.sleep(5)
                temp_sheet.Delete()
                pdfs.append(export_path)

            j += 1

        except Exception as e:
            print(f"Failed to process {file}: {e}")

        wb.Close(SaveChanges=False)
        excel.Quit()

        # crop pdf
        for pdf in pdfs:
            crop_pdf(pdf)