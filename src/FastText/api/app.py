from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import fasttext
import time

app = FastAPI()

# Load the trained model
model = fasttext.load_model("../training/ots_sms_model_v1.1.bin")

class SMS(BaseModel):
    text: str

@app.post("/predict/")
async def predict_sms(sms: SMS):
    start_time = time.time()  # Start time

    text = sms.text
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    label, probability = model.predict(text)

    end_time = time.time()  # End time
    processing_time = end_time - start_time  # Calculate processing time

    # Clean up the label format
    clean_label = label[0].replace('__label__', '')
    return {"label": clean_label, "probability": probability[0], "processing_time": processing_time}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
