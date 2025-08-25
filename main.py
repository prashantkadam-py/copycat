from fastapi import FastAPI
from pydantic import BaseModel
import copycat

app = FastAPI()

# Define request body schema
class CopycatRequest(BaseModel):
    brand: str
    ad_text: str


@app.post("/generate")
def generate(req: CopycatRequest):
   result = copycat.run(
                brand=req.brand,
                copy=req.ad_text,
                output_types=req.output_types
                )
   return {"brand": req.brand, "ad_text": req.ad_text, "result": result}



