from google import genai
from google.genai import types

client = genai.Client()

with open("plant.jpg", "rb") as f:
    image_data = f.read()

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=[
        types.Part.from_bytes(
            data=image_data,
            mime_type="image/jpeg"
        ),
        """Проанализируй фотографию растения.

Ответь по-русски и укажи:
1. Название растения.
2. Что видно на фотографии.
3. Есть ли признаки проблемы или болезни.
4. Наиболее вероятная причина.
5. Что рекомендуется сделать.
6. Насколько ты уверен в диагнозе: от 0 до 100%.

Если по одной фотографии нельзя поставить точный диагноз,
обязательно скажи об этом."""
    ]
)

print(response.text)