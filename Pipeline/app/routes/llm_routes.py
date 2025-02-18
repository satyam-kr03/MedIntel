from fastapi import APIRouter
from app.utils.llm_pipeline import generate_llm_response

router = APIRouter()

@router.post("/generate")
async def generate_output(inp: str):
    response = generate_llm_response(inp)
    return {"message": response}
