import pandas as pd
import requests
import time
from bs4 import BeautifulSoup

def scrap_volumes(link_to_main_page='https://portal.sinteza.singidunum.ac.rs/'):
    try:
        session = requests.Session()

        # original page to establish session
        response = session.get(link_to_main_page)
        response.raise_for_status() # Check for errors

        # parsing
        page = BeautifulSoup(response.text, 'html.parser')
        print('returning list of volumes')
        return [volume['href'] for volume in page.find_all('a', class_='volumes')]
    except Exception as e:
        print(e)
        return None

def scrap_papers(link_to_volume):
    try:
        session = requests.Session()

        # original page to establish session
        response = session.get(link_to_volume)
        response.raise_for_status() # Check for errors

        # parsing
        page = BeautifulSoup(response.text, 'html.parser')
        papers = page.find_all('tr')[1:]
        papers = [paper.find_all('td')[1].a for paper in papers]
        print('returning', link_to_volume)
        return [(paper['href'], paper.string) for paper in papers]
        
    except Exception as e:
        print(e)
        return None

def scrap_abstract(link_to_paper):
    try:
        session = requests.Session()

        # original page to establish session
        response = session.get(link_to_paper)
        response.raise_for_status() # Check for errors

        # parsing
        page = BeautifulSoup(response.text, 'html.parser')
        return page.div.find_all('div', recursive=False)[2].find_all('div', recursive=False)[2].find_all('div', recursive=False)[4].contents[6].strip()
        

    except Exception as e:
        print(e)
        return None
    
def get_all_papers(link_to_main_page='https://portal.sinteza.singidunum.ac.rs/'):
    volume_links = scrap_volumes(link_to_main_page)
    
    all_papers = []

    for volume_link in volume_links:
        all_papers.extend(scrap_papers(volume_link))
        time.sleep(0.5)
    
    return pd.DataFrame(all_papers, columns=['link', 'title'])

if __name__ == "__main__":
    # volumes = scrap_volumes('https://portal.sinteza.singidunum.ac.rs/')
    # print(volumes)
    df = get_all_papers()
    print(df.head(5))
    print(df.count())

    # papers = scrap_papers('https://portal.sinteza.singidunum.ac.rs/issue/showAll/2025')
    # print(papers)

    # scrap_abstract('https://portal.sinteza.singidunum.ac.rs/paper/920')