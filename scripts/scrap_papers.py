import time
import numpy as np
import requests
import json
from bs4 import BeautifulSoup
from google import genai

import constants
import manage_db as db
import manage_faiss as faiss


def get_kaznu_paper(path, paper_id, language_code = 'ru_RU'):
    try:
        # Target URL and desired language
        #url = f"https://bm.kaznu.kz/index.php/kaznu/article/view/{paper_id}"
        paper_url = path + 'article/view/' + str(paper_id)
        print(paper_url)

        # Create a session to maintain cookies
        session = requests.Session()

        # First, get the original page to establish session
        response = session.get(paper_url)
        response.raise_for_status() # Check for errors

        # Construct the language switch URL
        switch_url = path + 'user/setLocale/' + language_code
        params = {
            "source": f"{path}{paper_id}"
        }

        # Send the language switch request
        switch_response = session.post(switch_url, params=params)
        switch_response.raise_for_status()

        # Now get the page in the selected language
        localized_response = session.get(paper_url)
        localized_response.raise_for_status()

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(localized_response.text, 'html.parser')

        # title, abstract, citation
        title = soup.find('h1', class_='page_title').get_text(strip=True)
        abstract_element = (soup.find('section', class_='abstract'))
        # for br in abstract_element.find_all('br'):
        #     br.replace_with('\n')
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
            vector_id = faiss.add_to_faiss(embedding)
            db.add_paper(journal_id, paper_id, title, abstract, citation, vector_id)




if __name__ == "__main__":
    gai = genai.Client(api_key=constants.GEMINI_API_KEY)
    scrap_kaznu_journal('https://philart.kaznu.kz/index.php/1-FIL/', 4900, 4959, gai)
    # dict_to_json = {}
    # paper_list = []
    #
    # for current_paper_id in range(1300, 1700):
    #     current_data = get_math_paper(current_paper_id)
    #     if (current_data != None):
    #         paper_list.append(current_data)
    #         print(f"Paper {current_paper_id} added")
    #     else:
    #         print(f"Paper {current_paper_id} not found")
    #     time.sleep(0.5)
    #
    # dict_to_json['papers'] = paper_list
    #
    # with open('../../data/json/math.json', "w", encoding='utf-8') as f:
    #     json.dump(dict_to_json, f, ensure_ascii=False)


# Add your specific scraping logic here for other elements
# For example, to get the article content:
# article_content = soup.find('div', {'id': 'content'})  # Adjust selector as needed

# soup = BeautifulSoup(response.text, 'html.parser')

# def get_content(tag_name, property_name, property_value, output_property):
#     tags = soup.find_all(tag_name, attrs={property_name: property_value})
#     output = []
#     for tag in tags:
#         output.append(tag[output_property])
#     return output

# meta_description = soup.find_all('meta', attrs={'name': 'DC.Description'})
#
# print(meta_description)