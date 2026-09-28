from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from google import genai
from google.genai import types
import json
import time
from collections import defaultdict, deque

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_buckets = defaultdict(deque)

def enforce_rate_limit(request: Request):
    now = time.time()
    key = request.client.host if request.client else "unknown"
    bucket = _rate_buckets[key]
    while bucket and now - bucket[0] >= RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")
    bucket.append(now)

def image_bytes_match_type(data, mime_type):
    if mime_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False

def validate_analyze_payload(data):
    if not isinstance(data, dict):
        return False
    if not isinstance(data.get("plant_name"), str):
        return False
    if not isinstance(data.get("observations"), list) or not all(isinstance(x, str) for x in data["observations"]):
        return False
    if not isinstance(data.get("preliminary_problem"), str):
        return False
    causes = data.get("possible_causes")
    if not isinstance(causes, list) or not all(isinstance(x, dict) and isinstance(x.get("cause"), str) and isinstance(x.get("why"), str) for x in causes):
        return False
    confidence = data.get("confidence")
    return isinstance(confidence, int) and not isinstance(confidence, bool) and 0 <= confidence <= 100

def validate_consult_payload(data):
    if not isinstance(data, dict):
        return False
    string_fields = ("assessment", "main_hypothesis", "escalation")
    list_fields = ("facts", "alternative_causes", "reasoning", "action_plan", "avoid", "monitor")
    if any(not isinstance(data.get(k), str) for k in string_fields):
        return False
    if any(not isinstance(data.get(k), list) or not all(isinstance(x, str) for x in data[k]) for k in list_fields):
        return False
    days = data.get("follow_up_days")
    confidence = data.get("confidence")
    return (isinstance(days, int) and not isinstance(days, bool) and 1 <= days <= 365 and
            isinstance(confidence, int) and not isinstance(confidence, bool) and 0 <= confidence <= 100)

app = FastAPI(title="PLANTIVA", description="Персональная помощь комнатным растениям")
client = genai.Client()
MODEL = "gemini-3.6-flash"
MAX_ATTEMPTS = 3


def gemini_generate(contents):
    """Gemini only. Retry failures up to three total attempts and log diagnostics."""
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        attempt_number = attempt + 1
        print(f"[PLANTIVA] Gemini attempt {attempt_number}/{MAX_ATTEMPTS}", flush=True)

        try:
            response = client.models.generate_content(model=MODEL, contents=contents)
            print(
                f"[PLANTIVA] Gemini attempt {attempt_number}/{MAX_ATTEMPTS}: response received",
                flush=True,
            )
            return response
        except Exception as exc:
            last_error = exc
            error_text = str(exc)

            print(
                f"[PLANTIVA] Gemini attempt {attempt_number}/{MAX_ATTEMPTS} failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            # A 429 means Gemini has refused the request because a quota/rate
            # limit was reached. Retrying immediately would only create
            # unnecessary requests, especially for the free-tier daily quota.
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                print(
                    "[PLANTIVA] Gemini quota/rate limit reached (429). "
                    "Not retrying this request.",
                    flush=True,
                )
                raise

            if attempt < MAX_ATTEMPTS - 1:
                delay = 1.5 * (attempt + 1)
                print(f"[PLANTIVA] Retrying in {delay:.1f} seconds...", flush=True)
                time.sleep(delay)

    print("[PLANTIVA] All Gemini attempts failed.", flush=True)
    raise last_error


@app.get("/")
def root():
    return RedirectResponse(url="/ru/", status_code=307)


@app.get("/ru/")
def ru_home():
    return RedirectResponse(url="/ru/diagnosis/", status_code=307)


@app.get("/ru/diagnosis/")
def ru_diagnosis():
    return FileResponse("index.html")


@app.get("/app")
def legacy_app():
    return RedirectResponse(url="/ru/diagnosis/", status_code=307)


@app.get("/ru/problems/yellow-leaves/")
def yellow_leaves():
    return FileResponse("yellow_leaves_ru.html")


@app.post("/analyze")
async def analyze_plant(request: Request, file: UploadFile = File(...), initial_problem: str = Form("")):
    enforce_rate_limit(request)
    mime_type = (file.content_type or "").lower()
    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="unsupported_image_type")
    image_data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_data:
        raise HTTPException(status_code=400, detail="empty_upload")
    if len(image_data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image_too_large")
    if not image_bytes_match_type(image_data, mime_type):
        raise HTTPException(status_code=415, detail="invalid_image_content")
    context = (
        f'Пользователь пришёл с темой: "{initial_problem}". Учитывай этот контекст, '
        "но не считай причину заранее установленной."
        if initial_problem else
        "Пользователь начал общую диагностику без заранее указанной проблемы."
    )

    prompt = f"""
You are an experienced houseplant health consultant. {context}
Analyze the photo. This is STEP 1 of a two-step consultation.

DIAGNOSTIC RULES:
- Separate what is actually visible from hypotheses.
- Never present a suspected cause as an established fact.
- If the photo is insufficient, say so.
- Ask only questions that can materially distinguish plausible causes.
- Ask 1 to 4 questions, not a fixed number.
- Each question must contain ONE idea and normally be no more than 8-12 Russian words.
- Use ordinary language. Do not ask for things a normal user cannot reasonably observe
  (for example whether soil is dry "to the bottom" of the pot).
- Prefer single-choice answers over free text.
- Use type "single_choice" whenever sensible.
- Use type "text" only when predefined options would lose important information.
- For single_choice, provide 2 to 5 short options.
- Add an uncertainty option ("Не знаю", "Не помню") ONLY when a user could genuinely
  not know or remember the answer. Do not add it mechanically.
- Do not duplicate information already clearly visible in the photo.
- Confidence reflects the quality of available evidence, not decorative certainty.

Return ONLY valid JSON without Markdown, exactly in this structure:
{{
  "plant_name": "название растения или 'Не удалось уверенно определить'",
  "observations": ["наблюдаемый факт 1", "наблюдаемый факт 2"],
  "preliminary_problem": "осторожная предварительная оценка",
  "possible_causes": [
    {{
      "cause": "возможная причина",
      "why": "какие наблюдения делают её правдоподобной"
    }}
  ],
  "questions": [
    {{
      "id": "q1",
      "text": "короткий вопрос",
      "type": "single_choice",
      "options": ["вариант 1", "вариант 2", "вариант 3"]
    }}
  ],
  "confidence": 0
}}

For a text question use:
{{
  "id": "q1",
  "text": "короткий вопрос",
  "type": "text",
  "options": []
}}

Return 1-4 questions. possible_causes should normally contain 1-3 plausible causes.
Write all user-facing values in Russian. confidence must be 0-100.
"""

    try:
        response = gemini_generate([
            types.Part.from_bytes(data=image_data, mime_type=mime_type), prompt
        ])
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as exc:
            print(
                f"[PLANTIVA] /analyze JSON decode error: {exc}. "
                f"Gemini response: {response.text!r}",
                flush=True,
            )
            raise HTTPException(status_code=502, detail="invalid_ai_response")

        if not validate_analyze_payload(data):
            print("[PLANTIVA] /analyze invalid response schema.", flush=True)
            raise HTTPException(status_code=502, detail="invalid_ai_response")

        questions = data.get("questions")
        if not isinstance(questions, list) or not 1 <= len(questions) <= 4:
            print("[PLANTIVA] /analyze invalid questions payload.", flush=True)
            raise HTTPException(status_code=502, detail="invalid_ai_response")

        for index, question in enumerate(questions, start=1):
            if not isinstance(question, dict) or not question.get("text"):
                raise HTTPException(status_code=502, detail="invalid_ai_response")
            question["id"] = question.get("id") or f"q{index}"
            qtype = question.get("type", "single_choice")
            if qtype not in ("single_choice", "text"):
                qtype = "text"
            question["type"] = qtype
            options = question.get("options", [])
            question["options"] = options if isinstance(options, list) else []
            if qtype == "single_choice" and len(question["options"]) < 2:
                question["type"] = "text"
                question["options"] = []

        return data

    except HTTPException:
        raise
    except Exception as exc:
        print(
            f"[PLANTIVA] /analyze failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise HTTPException(status_code=503, detail="service_temporarily_unavailable")


@app.post("/consult")
async def consult(
    request: Request,
    plant_name: str = Form(...),
    observations: str = Form(...),
    preliminary_problem: str = Form(...),
    possible_causes: str = Form("[]"),
    qa_json: str = Form("[]"),
):
    enforce_rate_limit(request)
    try:
        observations_data = json.loads(observations)
    except Exception:
        observations_data = [observations] if observations else []

    try:
        causes_data = json.loads(possible_causes)
    except Exception:
        causes_data = []

    try:
        qa_data = json.loads(qa_json)
    except Exception:
        qa_data = []

    prompt = f"""
You are an experienced houseplant health consultant. This is STEP 2 of a two-step consultation.

PLANT: {plant_name}
OBSERVED FROM PHOTO: {json.dumps(observations_data, ensure_ascii=False)}
PRELIMINARY ASSESSMENT: {preliminary_problem}
PRELIMINARY HYPOTHESES: {json.dumps(causes_data, ensure_ascii=False)}
USER ANSWERS: {json.dumps(qa_data, ensure_ascii=False)}

DIAGNOSTIC RULES:
1. Keep three categories separate:
   A) facts visible in the photo,
   B) facts explicitly reported by the user,
   C) hypotheses/inferences.
2. Never turn a hypothesis into a fact later in the answer.
3. If more than one cause remains plausible, give the main hypothesis plus alternatives.
4. Prefer safe, reversible first actions.
5. Do NOT recommend invasive actions such as unpotting, cutting roots, cutting tissue,
   pesticides/fungicides or major repotting unless the available evidence specifically
   supports that intervention. If such action may become necessary, first explain what
   sign would justify it.
6. Do not invent temperatures, watering frequency, drainage quality, root condition,
   pests or environmental events that were not observed or reported.
7. Confidence must reflect evidence quality. Lower it when important information is missing.
8. Recommendations should be practical, calm and concise.

Return ONLY valid JSON without Markdown, exactly:
{{
  "assessment": "уточнённая, но осторожная оценка",
  "facts": [
    "факт из фото или ответа пользователя"
  ],
  "main_hypothesis": "основная вероятная причина",
  "alternative_causes": [
    "другая остающаяся возможной причина"
  ],
  "reasoning": [
    "почему гипотеза согласуется с фактами"
  ],
  "action_plan": [
    "безопасный и конкретный шаг"
  ],
  "avoid": [
    "чего сейчас не делать"
  ],
  "monitor": [
    "что наблюдать дальше"
  ],
  "escalation": "какой признак потребует более серьёзной проверки/действия",
  "follow_up_days": 7,
  "confidence": 0
}}

Use 2-5 action steps, only as many as useful.
alternative_causes may be empty when evidence is unusually clear.
Write all user-facing values in Russian.
follow_up_days must be an integer and confidence 0-100.
"""

    try:
        response = gemini_generate(prompt)
        try:
            data = json.loads(response.text)
            if not validate_consult_payload(data):
                print("[PLANTIVA] /consult invalid response schema.", flush=True)
                raise HTTPException(status_code=502, detail="invalid_ai_response")
            return data
        except json.JSONDecodeError as exc:
            print(
                f"[PLANTIVA] /consult JSON decode error: {exc}. "
                f"Gemini response: {response.text!r}",
                flush=True,
            )
            raise HTTPException(status_code=502, detail="invalid_ai_response")
    except HTTPException:
        raise
    except Exception as exc:
        print(
            f"[PLANTIVA] /consult failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise HTTPException(status_code=503, detail="service_temporarily_unavailable")
