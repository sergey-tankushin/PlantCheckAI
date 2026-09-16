from fastapi import FastAPI, UploadFile, File
from google import genai
from google.genai import types
import json

app = FastAPI(
    title="PlantCheck AI",
    description="AI-сервис для анализа состояния растений по фотографии"
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
Проанализируй фотографию растения.

Верни ТОЛЬКО корректный JSON без Markdown и без ```.

Формат ответа:

{
  "plant_name": "название растения",
  "observations": "что видно на фотографии",
  "problem": "признаки проблемы или болезни",
  "cause": "наиболее вероятная причина",
  "recommendation": "что рекомендуется сделать",
  "confidence": 0
}

Поле confidence должно содержать число от 0 до 100.

Если по одной фотографии нельзя поставить точный диагноз,
укажи это в соответствующих полях.
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