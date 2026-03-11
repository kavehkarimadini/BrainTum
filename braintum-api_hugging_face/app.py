from fastapi import FastAPI
import torch
import numpy as np
from pydantic import BaseModel

from model import UNetWithClassifier

app = FastAPI()

device="cpu"

model=UNetWithClassifier()
model.load_state_dict(torch.load("best_model_brain.pth",map_location=device))
model.eval()


class MRIInput(BaseModel):
    image:list


@app.post("/predict")

def predict(data:MRIInput):

    image=np.array(data.image)

    tensor=torch.tensor(image).float()

    with torch.no_grad():

        seg,cls=model(tensor)

        pred=torch.argmax(seg,dim=1).cpu().numpy()

    return {
        "mask":pred.tolist(),
        "tumor_probability":float(cls.item())
    }