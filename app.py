from app import create_app
from config import AppConfig
import os

app = create_app()

if __name__ == '__main__':
    # Run the Flask application with settings from config
    app.run(
        host=AppConfig.HOST,
        port=AppConfig.PORT,
        debug=AppConfig.DEBUG
    ) 