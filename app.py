from fastapi import FastAPI, UploadFile, File
from google import genai
from google.genai import types
import json

app = FastAPI(
    title="PlantCheck AI",
    description="AI service for plant analysis from a photo"
)

client = genai.Client()


@app.get("/")
def home():
    return {"message": "PlantCheck AI is working!"}


@app.post("/analyze")
async def analyze_plant(file: UploadFile = File(...)):
    image_data = await file.read()
    mime_type = file.content_type or "image/jpeg"

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(
                data=image_data,
                mime_type=mime_type
            ),
            """
Analyze the plant in the photo.

Return ONLY valid JSON without Markdown.

Use this structure:
{
  "plant_name": "plant name",
  "observations": "what is visible in the photo",
  "problem": "signs of a problem or disease",
  "cause": "most likely cause",
  "recommendation": "what should be done",
  "confidence": 0
}

Write the values in Russian.
The confidence field must be a number from 0 to 100.
If an exact diagnosis cannot be made from one photo, say so.
"""
        ]
    )

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError:
        return {
            "filename": file.filename,
            "analysis": response.text
        }

    return {
        "filename": file.filename,
        **result
    }

from fastapi.responses import FileResponse


@app.get("/app")
def web_app():
    return FileResponse("index.html")
