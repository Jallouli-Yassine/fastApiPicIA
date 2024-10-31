from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],  # Replace with your Laravel app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


# Define Pydantic model for request body
class ImageRequest(BaseModel):
    image_url: str

@app.post('/caption')
async def caption_image(request: ImageRequest):
    try:
        # Fetch the image from the URL
        response = requests.get(request.image_url)
        response.raise_for_status()  # Check for HTTP errors
        image = Image.open(io.BytesIO(response.content)).convert('RGB')

        # Process the image for the model
        inputs = processor(image, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Return the generated caption
        return {"caption": caption}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating caption: {str(e)}")

# Optional: Root endpoint to check if the API is working
@app.get("/")
async def root():
    return {"message": "Hello World"}


# Optional: Greeting endpoint
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


# Run the FastAPI app
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
