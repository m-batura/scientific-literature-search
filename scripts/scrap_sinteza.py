import numpy as np
import requests
from bs4 import BeautifulSoup

def scrap_volumes(link):
    try:
        session = requests.Session()

        # original page to establish session
        response = session.get(link)
        response.raise_for_status() # Check for errors

        # parsing
        soup = BeautifulSoup(response.text, 'html.parser')

        # return numpy array of links to volumes

    except requests.exceptions.RequestException as e:
        print(e)
        return None

    except Exception as e:
        print(e)
        return None