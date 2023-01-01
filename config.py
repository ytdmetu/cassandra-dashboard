import os 

class Config:
    BASE_URL = os.environ.get('BASE_URL', 'http://127.0.0.1:8000')
    FORECAST_INPUT_START_OFFSET = 14
    API_USERNAME = os.environ.get('API_USERNAME', 'api')
    API_PASSWORD = os.environ.get('API_PASSWORD', 'api_password123')