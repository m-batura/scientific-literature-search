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
        page = BeautifulSoup(response.text, 'html.parser')
        return np.array([volume['href'] for volume in page.find_all('a', class_='volumes')])

    except Exception as e:
        print(e)
        return None

def scrap_papers(link):
    try:
        session = requests.Session()

        # original page to establish session
        response = session.get(link)
        response.raise_for_status() # Check for errors

        # parsing
        page = BeautifulSoup(response.text, 'html.parser')
        # write return of 2d array [name, ref]

    except Exception as e:
        print(e)
        return None

if __name__ == "__main__":
    volumes = scrap_volumes('https://portal.sinteza.singidunum.ac.rs/')
    print(volumes)