from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from tempfile import NamedTemporaryFile
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
import base64
import os
import uvicorn
from ollama import Client
import json

# Initialize FastAPI application
app = FastAPI()

# Configure Llama client
HOST_URL = "http://3.136.141.248:8000"
client = Client(host=HOST_URL)

# Llama model configuration
LLAMA_MODEL = 'llama3.2-vision:90b-instruct-q8_0'

def convert_image_to_base64(image_path: str) -> str:
    """Convert an image to a base64-encoded string."""
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    return base64.b64encode(image_bytes).decode("utf-8")


def pdf_to_images(pdf_file: UploadFile) -> str:
    """Convert PDF pages to images and return the first page image path."""
    try:
        # Save the uploaded PDF temporarily
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.file.read())
            temp_pdf_path = temp_pdf.name

        # Convert PDF to images
        images = convert_from_path(temp_pdf_path, fmt='png')
        if not images:
            raise HTTPException(status_code=400, detail="Failed to convert PDF to images.")

        # Use the first page image for processing
        first_page_image = images[0]
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            first_page_image.save(temp_img.name, format="PNG")
            return temp_img.name
    finally:
        # Cleanup temporary PDF file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


def process_with_llama(image_path: str) -> str:
    """Process an image with the Llama model."""
    image_base64 = convert_image_to_base64(image_path)
    response = client.chat(
        model=LLAMA_MODEL,
        messages=[

            {"role": "system",
             "content": """
                        You are a specialist in comprehending receipts.
                        Input images in the form of receipts will be provided to you,
                        and your task is to respond to questions based on the content of the input image.
                        """
             },
            {"role": "user",
             "content": """
                        Convert Invoice data into JSON format with appropriate JSON tags as 
                        required for the data in the image and  extracting all items listed 
                        in the receipt.
                        Provide only the transcription without any additional comments.
                        """,
             "images": [image_base64]
             }
        ]
    )
    overall_summary = response['message']['content']

    # to parse into Json Response
    json_start_index = overall_summary.find("{")
    json_end_index = overall_summary.rfind("}")
    if json_start_index != -1 and json_end_index != -1:
        overall_summary_json = overall_summary[json_start_index:json_end_index + 1]
        return json.loads(overall_summary_json)
    else:
        raise ValueError("Invalid Google Flash response format")







@app.post("/process-receipt/")
async def process_receipt(file: UploadFile = File(...)):
    """
    Endpoint to process receipt images and return extracted data in JSON format.
    """

    try:

        

        # Check file type and process accordingly
        if file.filename.endswith(".pdf"):
            # Convert PDF to image
            image_path = pdf_to_images(file)
        elif file.filename.endswith((".png", ".jpeg", ".jpg", ".webp")):
            # Save the uploaded image temporarily
            with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_img:
                temp_img.write(file.file.read())
                image_path = temp_img.name
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        # Process the image with Llama model
        llama_response = process_with_llama(image_path)

        # Cleanup temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)

        return llama_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
