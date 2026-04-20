import time
import requests
from bs4 import BeautifulSoup


def get_infom_paper(path, paper_id, language_code = 'en_US'):
    try:
        # common_path = path# + '/index.php/1-FIL/'
        paper_url = path + 'article/view/' + str(paper_id)
        print(paper_url)

        # session to maintain cookies
        session = requests.Session()

        # original page to establish session
        response = session.get(paper_url)
        response.raise_for_status() # Check for errors

        # language switch URL
        switch_url = path + 'user/setLocale/' + language_code
        params = {
            "source": f"{path}{paper_id}"
        }

        # language switch request
        switch_response = session.post(switch_url, params=params)
        switch_response.raise_for_status()

        # page in the selected language
        localized_response = session.get(paper_url)
        localized_response.raise_for_status()

        # parsing
        soup = BeautifulSoup(localized_response.text, 'html.parser')

        # print(soup)

        # extraction of title, abstract, citation
        title = soup.find('h1', class_='page_title').get_text(strip=True)
        abstract_element = (soup.find('div', class_='item abstract'))
        all_paragraphs = abstract_element.find_all('p')
        abstract = '\n'.join([
            paragraph.get_text() for paragraph in all_paragraphs])

        return [title, abstract]

    except requests.exceptions.RequestException as e:
        print(e)
        time.sleep(0.5)
        return None

    except Exception as e:
        print(e)
        time.sleep(0.5)
        return None
    
if __name__ == "__main__":
    print(get_infom_paper('https://www.infom.org.rs/index.php/infom/', 2709, 'en_US'))