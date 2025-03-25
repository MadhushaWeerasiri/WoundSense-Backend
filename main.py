from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    MODEL = tf.keras.models.load_model("../model/1.h5")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    MODEL = None

CLASS_NAMES = ['Abrasions', 'Bruises', 'Burns', 'Cut', 'Diabetic', 'Laceration', 'Normal', 'Pressure', 'Surgical', 'Venous']
SUGESSIONS = [
    {
        "type": "Abrasions",
        "suggestions": [
            "Clean the area gently with mild soap and water to remove dirt and debris.",
            "Apply an antiseptic ointment (e.g., Neosporin) to prevent infection.",
            "Cover with a sterile bandage or adhesive bandage to keep it clean.",
            "Change the dressing daily or if it becomes wet or dirty.",
            "Monitor for signs of infection (redness, swelling, pus)."
        ]
    },
    {
        "type": "Bruises",
        "suggestions": [
            "Apply a cold pack or ice wrapped in a cloth for 10-15 minutes to reduce swelling and pain (do this within the first 24-48 hours).",
            "Elevate the affected area if possible to minimize blood flow.",
            "After 48 hours, switch to a warm compress to help the body reabsorb blood.",
            "Avoid pressing on the bruise to prevent further damage.",
            "Seek medical attention if the bruise is unexplained, severe, or doesn’t fade after 2 weeks."
        ]
    },
    {
        "type": "Burns",
        "suggestions": [
            "For minor (first-degree) burns: Run cool (not cold) water over the area for 10-15 minutes.",
            "Do not apply ice, butter, or oily substances.",
            "Apply aloe vera or a burn ointment and cover with a sterile, non-stick dressing.",
            "For second-degree (blisters) or third-degree (deep tissue) burns: Seek immediate medical attention—do not attempt to treat at home beyond cooling and covering loosely.",
            "Watch for infection or worsening pain."
        ]
    },
    {
        "type": "Cut",
        "suggestions": [
            "Stop bleeding by applying gentle pressure with a clean cloth or bandage.",
            "Rinse the cut under running water and clean with mild soap (avoid getting soap in the wound).",
            "Apply an antibiotic ointment and cover with a sterile bandage.",
            "Change the dressing daily and keep the wound dry.",
            "Seek medical help if the cut is deep, won’t stop bleeding after 10 minutes, or shows signs of infection."
        ]
    },
    {
        "type": "Diabetic",
        "suggestions": [
            "Do not attempt to treat alone—consult a healthcare provider immediately.",
            "Keep the wound clean and dry; avoid soaking the foot unless instructed.",
            "Apply dressings as prescribed by a doctor.",
            "Avoid putting weight on the affected area (use crutches or a wheelchair if needed).",
            "Monitor blood sugar levels closely, as high glucose delays healing."
        ]
    },
    {
        "type": "Laceration",
        "suggestions": [
            "Apply pressure with a clean cloth to stop bleeding.",
            "Rinse with water to remove debris, but do not scrub.",
            "Cover with a sterile bandage or cloth; seek medical attention if it’s deep or wide.",
            "Stitches may be needed within 6-8 hours—don’t delay if unsure.",
            "Watch for infection or if debris remains trapped."
        ]
    },
    {
        "type": "Normal",
        "suggestions": [
            "Keep skin clean with regular washing.",
            "Moisturize to prevent dryness and cracking.",
            "Protect from injury with appropriate clothing or gear.",
            "No specific treatment needed unless it becomes damaged."
        ]
    },
    {
        "type": "Pressure",
        "suggestions": [
            "Relieve pressure by repositioning the person every 1-2 hours.",
            "Clean gently with saline or mild soap and water; pat dry.",
            "Apply a protective dressing as advised by a healthcare provider.",
            "Use cushions or special mattresses to reduce pressure.",
            "Seek medical help for deep or worsening sores."
        ]
    },
    {
        "type": "Surgical",
        "suggestions": [
            "Follow your surgeon’s specific care instructions.",
            "Keep the area clean and dry; avoid soaking (e.g., baths) until healed or cleared by a doctor.",
            "Change dressings as directed, usually daily or if wet.",
            "Avoid picking at stitches or staples.",
            "Report signs of infection (redness, warmth, pus) or opening of the wound to your doctor."
        ]
    },
    {
        "type": "Venous",
        "suggestions": [
            "Consult a healthcare provider for proper management.",
            "Elevate the leg above heart level when resting to improve circulation.",
            "Keep the wound clean and apply dressings as prescribed (often moist dressings).",
            "Wear compression stockings or bandages if recommended.",
            "Monitor for increased swelling, odor, or pain, and report to a doctor."
        ]
    }
]

@app.get("/ping")
async def ping():
    return {"message": "Hello, FastAPI is running!"}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image = image.resize((240, 240))
        return np.array(image)
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

def get_suggestions(predicted_class: str) -> list:
    """Return the suggestions for the predicted wound type."""
    for suggestion in SUGESSIONS:
        if suggestion["type"] == predicted_class:
            return suggestion["suggestions"]
    return ["No specific suggestions available for this class."]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model failed to load.")

    logger.info(f"Received file: {file.filename} ({file.content_type})")

    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)

        prediction = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0]) * 100

        suggestions = get_suggestions(predicted_class)

        logger.info(f"Prediction: {predicted_class} ({confidence:.2f}%)")

        # Return prediction, confidence, and suggestions
        return {
            "Class": predicted_class,
            "Confidence": round(float(confidence), 2),
            "Suggestions": suggestions
        }

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image.")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)