import base64
import math
import os

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

SECRET = os.environ.get("SERVICE_SECRET", "")

app = FastAPI()


class Corners(BaseModel):
    TL: list[float]
    TR: list[float]
    BR: list[float]
    BL: list[float]


class WarpRequest(BaseModel):
    image_base64: str
    corners: Corners


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


@app.post("/warp")
def warp(body: WarpRequest, x_service_secret: str = Header(default="")):
    if SECRET and x_service_secret != SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    img_bytes = base64.b64decode(body.image_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    c = body.corners
    TL, TR, BR, BL = c.TL, c.TR, c.BR, c.BL
    out_w = int(max(_dist(TL, TR), _dist(BL, BR)))
    out_h = int(max(_dist(TL, BL), _dist(TR, BR)))

    src = np.float32([TL, TR, BR, BL])
    dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    H = cv2.getPerspectiveTransform(src, dst)
    corrected = cv2.warpPerspective(
        img,
        H,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    _, buf = cv2.imencode(".png", corrected)
    return {"image_base64": base64.b64encode(buf.tobytes()).decode()}


@app.get("/health")
def health():
    return {"ok": True}
