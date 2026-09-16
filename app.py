from fastapi import FastAPI, UploadFile, File
from google import genai
from google.genai import types

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
            """Проанализируй фотографию растения.

Ответь по-русски и укажи:
1. Название растения.
2. Что видно на фотографии.
3. Есть ли признаки проблемы или болезни.
4. Наиболее вероятная причина.
5. Что рекомендуется сделать.
6. Уверенность в диагнозе от 0 до 100%.

Если по одной фотографии нельзя поставить точный диагноз,
обязательно скажи об этом."""
        ]
    )

    return {
        "filename": file.filename,
        "analysis": response.text
    }