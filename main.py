from fastapi import FastAPI, HTTPException, UploadFile, File, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import time

# Import des modèles et services
from models import ImageRequest, TextProcessingRequest
from services.Service import Service

# Création de l'application
app = FastAPI()

# Création d'un routeur avec préfixe /fastapi pour correspondre à la configuration nginx
router = APIRouter(prefix="/fastapi")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permettre toutes les origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du service
service = Service()

@router.post('/process-text')
async def process_text(request: TextProcessingRequest):
    try:
        result = service.process_text(request.input_text)
        return {"output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@router.post('/caption')
async def caption(file: UploadFile = File(...)):
    try:
        caption = service.generate_caption(file.file)
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating caption: {str(e)}")

@router.post('/team-logo-description')
async def team_logo_description(team_name: str = Form(...), file: UploadFile = File(...)):
    try:
        description = service.generate_team_logo_description(file.file, team_name)
        return {"description": description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating team logo description: {str(e)}")

# Generate multiple suggestions for team logo descriptions PFE
@router.post('/team-logo-descriptions')
async def team_logo_descriptions(team_name: str = Form(...), file: UploadFile = File(...), num_suggestions: int = Form(3)):
    try:
        descriptions = service.generate_team_logo_descriptions(file.file, team_name, num_suggestions)
        return {"descriptions": descriptions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating team logo descriptions: {str(e)}")

# Optional: Root endpoint to check if the API is working
@router.get("/")
async def root():
    return {"message": "Hello World"}


# Optional: Greeting endpoint
@router.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

# Route racine qui redirige vers /fastapi
@app.get("/")
async def redirect_root():
    return {"message": "Redirect to /fastapi/"}

# Inclure le routeur dans l'application
app.include_router(router)

# Démarrer le serveur si le script est exécuté directement
if __name__ == "__main__":
    import uvicorn
    print("Démarrage du serveur FastAPI...")
    print("Accédez à l'API sur: http://localhost:8000")
    print("Pour la documentation de l'API, visitez: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
