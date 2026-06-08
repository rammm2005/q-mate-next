from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="CodeQ-Mate",
    description="Context-aware question answering system for internal software repositories",
    version="0.1.0",
)

# Register API routes
app.include_router(router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
