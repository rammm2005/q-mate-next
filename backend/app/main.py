from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="CodeQ-Mate",
    description="Context-aware question answering system for internal software repositories",
    version="0.1.0",
)

# CORS - allow frontend to call backend directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
