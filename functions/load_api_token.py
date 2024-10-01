import os
from dotenv import load_dotenv

def load_api_token():
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve the Hugging Face API token
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Check if the API token is set; if not, raise an error
    if api_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
    else:
        raise ValueError("Hugging Face API token is not set in the .env file.")
