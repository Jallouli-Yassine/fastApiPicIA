from pydantic import BaseModel

# Define Pydantic model for image request
class ImageRequest(BaseModel):
    image_url: str

# Define Pydantic model for text processing request
class TextProcessingRequest(BaseModel):
    input_text: str

