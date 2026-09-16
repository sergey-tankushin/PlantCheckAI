from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Ответь только одной фразой по-русски: PlantCheck AI подключен к Gemini."
)

print(response.text)