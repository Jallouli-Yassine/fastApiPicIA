from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

from fastapi.middleware.cors import CORSMiddleware
import io
import time
import torch
from fastapi import UploadFile, File


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Update with your Angular app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

modelText = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large").to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")



# Define Pydantic model for request body
class ImageRequest(BaseModel):
    image_url: str


# Define Pydantic model for request body
class TextProcessingRequest(BaseModel):
    input_text: str



@app.post('/process-text')
async def process_text(request: TextProcessingRequest):
    try:
        # Tokenize input text
        input_ids = tokenizer(request.input_text, return_tensors="pt").input_ids.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate output using the model
        outputs = modelText.generate(input_ids)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"output": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


@app.post('/caption')
async def caption(file: UploadFile = File(...)):
    try:
        # Start timer for processing
        start_time = time.time()

        # Process the image and generate a description
        textPastry = "pastry product of"
        textLaravel = "garden of"
        # place = "the location of the event is :"

        # Open the uploaded file as an image
        image = Image.open(file.file).convert('RGB')

        # Process the image for the model

        inputs = processor(image, textLaravel, return_tensors="pt").to(model.device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        print("cap : ",caption);
        # Log processing time
        elapsed_time = time.time() - start_time
        print(f"Processing time: {elapsed_time:.2f} seconds")

        # Return the generated caption
        return {"caption": caption}

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

    uvicorn.run(app, host="127.0.0.1", port=8081)  # Note: Ensure port matches the one used in Laravel
