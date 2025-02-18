from fastapi import APIRouter
from app.utils.pdf_processing import process_pdf

router = APIRouter()

@router.post("/upload")
async def post_pdf():
    try:
        raw_pdf_elements = process_pdf("uploaded_file.pdf", "./")
        return {"message": "200", "data": raw_pdf_elements}
    except Exception as e:
        print(e)
        return {"message": "404"}
