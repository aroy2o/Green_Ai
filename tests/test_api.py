
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_upload_valid_image():
    import pytest
    from PIL import Image
    img_path = "tests/sample1.jpg"
    if not os.path.exists(img_path):
        pytest.skip("sample1.jpg not found. Please add a valid JPEG image at tests/sample1.jpg to run this test.")
    # Check if file is a valid JPEG
    try:
        with Image.open(img_path) as im:
            if im.format not in ("JPEG", "JPG"):
                pytest.skip("sample1.jpg is not a valid JPEG image. Please replace it with a valid JPEG.")
    except Exception:
        pytest.skip("sample1.jpg is not a valid image. Please replace it with a valid JPEG.")
    with open(img_path, "rb") as img:
        response = client.post("/ocr", files={"file": ("sample1.jpg", img, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "fail"]
    assert "object_detection" in data
    if data["status"] == "success":
        assert "ocr" in data
        assert "full_text" in data["ocr"]
        assert "structured_text" in data["ocr"]

def test_upload_invalid_type():
    with open("tests/sample.txt", "rb") as f:
        response = client.post("/ocr", files={"file": ("sample.txt", f, "text/plain")})
    assert response.status_code == 400
    assert (
        "File too large" in response.json()["detail"]
        or "Corrupted image file" in response.json()["detail"]
        or "Please upload a valid JPG or PNG" in response.json()["detail"]
        or "File too large. Please upload an image" in response.json()["detail"]
    )

def test_upload_large_file():
    # Create a dummy large file
    with open("tests/large.jpg", "wb") as f:
        f.write(os.urandom(6 * 1024 * 1024))
    with open("tests/large.jpg", "rb") as f:
        response = client.post("/ocr", files={"file": ("large.jpg", f, "image/jpeg")})
    assert response.status_code == 400
    assert (
        "File too large" in response.json()["detail"]
        or "File too large. Please upload an image" in response.json()["detail"]
    )
    os.remove("tests/large.jpg")
