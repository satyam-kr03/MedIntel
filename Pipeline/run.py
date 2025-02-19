import uvicorn
from pyngrok import ngrok
from app.config import NGROK_AUTH_TOKEN
from app.main import app

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

if __name__ == "__main__":
    try:
        tunnel = ngrok.connect(8000)
        print(f"Ngrok tunnel established at: {tunnel.public_url}")
    except Exception as e:
        print(f"Error establishing ngrok tunnel: {e}")
        exit(1)

    uvicorn.run(app, host="0.0.0.0", port=8000)
