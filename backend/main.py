from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes_audio import router as audio_router
from api.routes_modes import router as modes_router
from api.routes_basis import router as basis_router
from api.routes_edge import router as edge_router
from api.routes_ai import router as ai_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create required directories on startup."""
    for directory in ["uploads", "outputs"]:
        os.makedirs(directory, exist_ok=True)
    yield


app = FastAPI(title="Signal Equalizer API", lifespan=lifespan)

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(audio_router)
app.include_router(modes_router)
app.include_router(basis_router)
app.include_router(edge_router)
app.include_router(ai_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)