import uvicorn
from .main import app
from ..utils.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

