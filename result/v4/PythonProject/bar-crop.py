import os
import win32com.client
import pymupdf  # PyMuPDF

file_path = r'C:\Users\mehrab\Desktop\QUEST\QUEST\result\v4\charts_data.xlsx'
output_folder = r'C:\Users\mehrab\Desktop\QUEST\QUEST\result\v4\results-bar'

sheet_names = [
    "AoI_Small", "AoI_Medium", "AoI_Large",
    "Energy_Small", "Energy_Medium", "Energy_Large",
    "Makespan_Small", "Makespan_Medium", "Makespan_Large"
]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def crop_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    for page in doc:
        page.set_cropbox(pymupdf.Rect(50, 315, 560, 475))

    temp_path = pdf_path.replace(".pdf", "_cropped.pdf")
    doc.save(temp_path, garbage=3, deflate=True)
    doc.close()
    os.remove(pdf_path)
    os.rename(temp_path, pdf_path)


def export_charts_clean():
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False

    try:
        wb = excel.Workbooks.Open(file_path)

        for name in sheet_names:
            try:
                ws = wb.Sheets(name)
                if ws.ChartObjects().Count > 0:
                    chart_obj = ws.ChartObjects(1)
                    print(f"Processing: {name}...")
                    export_path = os.path.join(output_folder, f"{name}.pdf")
                    temp_sheet = wb.Sheets.Add()
                    chart_obj.Copy()
                    temp_sheet.Paste()
                    excel.CutCopyMode = False
                    pasted_chart = temp_sheet.ChartObjects(1)
                    pasted_chart.Left = 0
                    pasted_chart.Top = 50
                    temp_sheet.ExportAsFixedFormat(0, export_path)

                    temp_sheet.Delete()

                    crop_pdf(export_path)
                    print(f"✓ Done: {name}")
                else:
                    print(f"⚠ No chart found in {name}")
            except Exception as e:
                print(f"❌ Error in {name}: {e}")

        wb.Close(SaveChanges=False)
    finally:
        excel.DisplayAlerts = True
        excel.Quit()


if __name__ == "__main__":
    export_charts_clean()