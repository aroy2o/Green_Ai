import os
from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_upload_valid_image():
    with open("tests/sample1.jpg", "rb") as img:
        response = client.post("/ocr", files={"file": ("sample1.jpg", img, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "fail"]
    assert "objects_detected" in data
    if data["status"] == "success":
        assert "full_text" in data
        assert "structured_text" in data

def test_upload_invalid_type():
    with open("tests/sample.txt", "rb") as f:
        response = client.post("/ocr", files={"file": ("sample.txt", f, "text/plain")})
    assert response.status_code == 400
    assert "Unsupported or corrupted image" in response.json()["detail"]

def test_upload_large_file():
    # Create a dummy large file
    with open("tests/large.jpg", "wb") as f:
        f.write(os.urandom(6 * 1024 * 1024))
    with open("tests/large.jpg", "rb") as f:
        response = client.post("/ocr", files={"file": ("large.jpg", f, "image/jpeg")})
    assert response.status_code == 400
    assert "File too large" in response.json()["detail"]
    os.remove("tests/large.jpg")
