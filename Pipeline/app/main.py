from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import image_routes, pdf_routes, llm_routes
from fastapi.responses import RedirectResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_routes.router, prefix="/image")
app.include_router(pdf_routes.router, prefix="/pdf")
app.include_router(llm_routes.router, prefix="/llm")

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")
