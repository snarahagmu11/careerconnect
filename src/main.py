from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router  

def create_app():
    app = FastAPI(title="careerconnect-backend", version="0.1.0")
    
    # Enable CORS for Streamlit frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # You can restrict this later
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Include all API routes
    app.include_router(router, prefix="/api")

    return app

app = create_app()
