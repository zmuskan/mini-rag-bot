from document_loader import load_document
print("TXT ->", load_document("sample.txt")[:400])

try:
    print("DOCX ->", load_document("data/sample.docx")[:400])
except Exception as e:
    print("DOCX -> error:", e)
try:
    print("PPTX ->", load_document("data/sample.pptx")[:400])
except Exception as e:
    print("PPTX -> error:", e)
try:
    print("PDF ->", load_document("data/sample.pdf")[:400])
except Exception as e:
    print("PDF -> skipped or error:", e)
