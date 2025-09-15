import time
import numpy as np
import requests
import json
from bs4 import BeautifulSoup
from google import genai

import constants
import sqlite_controller as db
import faiss_controller as faiss

# scraps paper from any kaznu journal
def get_kaznu_paper(path, paper_id, language_code = 'ru_RU'):
    try:
        common_path = path + '/index.php/1-FIL/'
        paper_url = common_path + 'article/view/' + str(paper_id)
        print(paper_url)

        # session to maintain cookies
        session = requests.Session()

        # original page to establish session
        response = session.get(paper_url)
        response.raise_for_status() # Check for errors

        # language switch URL
        switch_url = common_path + 'user/setLocale/' + language_code
        params = {
            "source": f"{common_path}{paper_id}"
        }

        # language switch request
        switch_response = session.post(switch_url, params=params)
        switch_response.raise_for_status()

        # page in the selected language
        localized_response = session.get(paper_url)
        localized_response.raise_for_status()

        # parsing
        soup = BeautifulSoup(localized_response.text, 'html.parser')

        # extraction of title, abstract, citation
        title = soup.find('h1', class_='page_title').get_text(strip=True)
        abstract_element = (soup.find('section', class_='abstract'))
        all_paragraphs = abstract_element.find_all('p')

        abstract = '\n'.join([
            paragraph.get_text() for paragraph in all_paragraphs])
        citation = soup.find('div', class_='csl-entry').get_text()

        return [title, abstract, citation]

    except requests.exceptions.RequestException as e:
        print(e)
        time.sleep(0.5)
        return None

    except Exception as e:
        print(e)
        time.sleep(0.5)
        return None

# get multiple papers in id range from one journal
def scrap_kaznu_journal(path, start_id, end_id, client):
    journal_id = db.check_journal_by_name(path)
    for paper_id in range(start_id, end_id + 1):
        if db.paper_exists(journal_id, paper_id):
            print(f"Skipping existing paper: journal_id={journal_id}, paper_id={paper_id}")
            continue

        paper = get_kaznu_paper(path, paper_id)
        if paper:
            title, abstract, citation = paper
            embedding_raw = faiss.get_embedding(abstract, client)
            embedding = np.array(embedding_raw, dtype='float32').reshape(1, -1)
            vector_id = faiss.add_to_faiss(embedding, papers_path)
            db.add_paper(journal_id, paper_id, title, abstract, citation, vector_id)




if __name__ == "__main__":
    gai = genai.Client(api_key=constants.GEMINI_API_KEY)
    scrap_kaznu_journal('https://philart.kaznu.kz', 4900, 4959, gai)