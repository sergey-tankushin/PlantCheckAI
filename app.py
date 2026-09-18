from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
import json

app = FastAPI(
    title="PlantCheck AI",
    description="AI plant health consultation"
)

client = genai.Client()


@app.get("/")
def home():
    return {"message": "PlantCheck AI is working!"}


@app.get("/app")
def web_app():
    return FileResponse("index.html")


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
You are an experienced plant health consultant.

Analyze the plant photo.

This is STEP 1 of a two-step consultation.

Do not make an overly confident diagnosis from one photo.

Return ONLY valid JSON without Markdown.

Use exactly this structure:

{
  "plant_name": "plant name",
  "observations": "what is visible",
  "preliminary_problem": "possible problem",
  "preliminary_cause": "most likely possible cause",
  "questions": [
    "question 1",
    "question 2",
    "question 3"
  ],
  "confidence": 0
}

Ask exactly 3 useful diagnostic questions.

Questions should depend on what you see in THIS photo.

Prefer questions about:
watering,
soil moisture,
drainage,
light,
temperature,
recent changes,
roots,
pests,
fertilizer,
or other factors relevant to the visible symptoms.

Do not ask something that can already be clearly seen in the photo.

Write all values in Russian.

confidence must be a number from 0 to 100.
"""
        ]
    )

    try:
        return json.loads(response.text)
    except Exception:
        return {
            "error": "AI returned an invalid response."
        }


@app.post("/consult")
async def consult(
    plant_name: str = Form(...),
    observations: str = Form(...),
    preliminary_problem: str = Form(...),
    preliminary_cause: str = Form(...),
    question1: str = Form(...),
    answer1: str = Form(...),
    question2: str = Form(...),
    answer2: str = Form(...),
    question3: str = Form(...),
    answer3: str = Form(...)
):

    prompt = f"""
You are an experienced plant health consultant.

This is STEP 2 of a plant consultation.

The first photo analysis produced:

Plant:
{plant_name}

Visible observations:
{observations}

Preliminary problem:
{preliminary_problem}

Preliminary possible cause:
{preliminary_cause}

The AI asked these diagnostic questions.

Question 1:
{question1}

User answer:
{answer1}

Question 2:
{question2}

User answer:
{answer2}

Question 3:
{question3}

User answer:
{answer3}

Use the original observations AND the user's answers.

Do not claim certainty when the available information is insufficient.

Return ONLY valid JSON without Markdown.

Use exactly this structure:

{{
  "diagnosis": "updated assessment of the problem",
  "cause": "most likely cause based on all available information",
  "action_plan": [
    "step 1",
    "step 2",
    "step 3",
    "step 4"
  ],
  "monitor": "what the user should observe during the next days",
  "follow_up_days": 7,
  "confidence": 0
}}

Give practical actions that the user can actually perform.

If dangerous symptoms such as severe root rot or a serious pest problem
may be present, explain what should be checked.

Write all values in Russian.

follow_up_days must be an integer.
confidence must be a number from 0 to 100.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    try:
        return json.loads(response.text)
    except Exception:
        return {
            "error": "AI returned an invalid response."
        }
