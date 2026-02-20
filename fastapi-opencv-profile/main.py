import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    # 1. Simpan Gambar
    image_data = await file.read()
    file_extension = file.filename.split(".")[-1]
    filename = f"{uuid4()}.{file_extension}"
    file_path = os.path.join("static", "uploads", filename)

    with open(file_path, "wb") as f:
        f.write(image_data)

    # 2. Proses Gambar dengan OpenCV
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    # Split Channel (OpenCV defaultnya BGR, kita pisah jadi B, G, R)
    b, g, r = cv2.split(img)

    # --- MODIFIKASI 1: Ambil Sampel Matrix (Misal 20x20 pixel pojok kiri atas) ---
    # Kita tidak mengirim seluruh gambar karena akan memberatkan browser
    limit = 20 
    matrix_r = r[:limit, :limit].tolist()
    matrix_g = g[:limit, :limit].tolist()
    matrix_b = b[:limit, :limit].tolist()

    # --- MODIFIKASI 2: Hitung Data Histogram ---
    # Menghitung distribusi frekuensi warna (0-255)
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256]).flatten().tolist()
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten().tolist()
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten().tolist()

    return templates.TemplateResponse("display.html", {
        "request": request,
        "image_path": f"/static/uploads/{filename}",
        "matrix_r": matrix_r,
        "matrix_g": matrix_g,
        "matrix_b": matrix_b,
        "hist_r": hist_r,
        "hist_g": hist_g,
        "hist_b": hist_b,
        "limit": limit
    })