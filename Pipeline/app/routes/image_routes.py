from fastapi import APIRouter
from app.utils.image_processing import image_to_base64
from app.dependencies import llava, retriever, id_key
from langchain.schema.document import Document
import uuid

router = APIRouter()

@router.post("/upload_img")
async def upload_image(inp: str):
    try:
        img_summary = []
        RES = llava(prompt="Describe the image...", images=[image_to_base64(f"./{inp}")])
        img_summary.append(RES)

        img_ids = [str(uuid.uuid4()) for _ in img_summary]
        summary_img = [Document(page_content=s, metadata={id_key: img_ids[i]}) for i, s in enumerate(img_summary)]

        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, img_summary)))

        return {"message": "200"}
    except Exception as e:
        print(e)
        return {"message": "404"}
