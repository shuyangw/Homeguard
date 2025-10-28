import os
from dotenv import load_dotenv

def load_apikey():
    load_dotenv()
    return {
        'api_key': os.getenv('API_KEY'),
        'api_secret': os.getenv('API_SECRET')
    }

# Load keys once when module is imported
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')