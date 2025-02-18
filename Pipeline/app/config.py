import os
import torch
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Tesseract configuration
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "C:/Users/go39res/AppData/Local/Programs/Tesseract-OCR")
os.environ["PATH"] += os.pathsep + TESSERACT_PATH

# Ngrok Configuration
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "2rr1W3HKR8YWgtA1UBoBuGI7vBM_4yPLBV8EibUReXoc6oV1d")

# Torch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
