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
        papers = page.find_all('tr')[1:]
        papers = [paper.find_all('td')[1].a for paper in papers]
        return np.array([[paper.string, paper['href']] for paper in papers])
        

    except Exception as e:
        print(e)
        return None

def scrap_abstract(link):
    try:
        session = requests.Session()

        # original page to establish session
        response = session.get(link)
        response.raise_for_status() # Check for errors

        # parsing
        page = BeautifulSoup(response.text, 'html.parser')
        print(page.div.find_all('div', recursive=False)[2].find_all('div', recursive=False)[2].find_all('div', recursive=False)[4].contents[6].strip())
        

    except Exception as e:
        print(e)
        return None

if __name__ == "__main__":
    # volumes = scrap_volumes('https://portal.sinteza.singidunum.ac.rs/')
    # print(volumes)

    papers = scrap_papers('https://portal.sinteza.singidunum.ac.rs/issue/showAll/2025')
    print(papers)

    # scrap_abstract('https://portal.sinteza.singidunum.ac.rs/paper/920')