import sqlite3
import json

import faiss_controller
from scripts import gai_controller

db_path = '../data/db/db.sqlite'

# displays all created tables
def fetch_all_tables():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''SELECT name FROM sqlite_master WHERE type='table';''')
    print(cursor.fetchall())

# displays table columns
def view_table(table):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()

    for column in columns:
        print(column)

# adds journal to journals table
def add_journal(name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Step 1: Try to insert the journal name (will fail silently if it already exists)
    try:
        cursor.execute("INSERT INTO journals (name) VALUES (?)", (name,))
        conn.commit()
    except sqlite3.IntegrityError:
        print("Journal already exists")
        pass  # Journal already exists due to UNIQUE constraint

    conn.close()

# adds kaznu paper to papers
def add_kaznu_paper(journal_id, paper_id, title, annotation, citation, vector_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute('''
                            INSERT INTO papers (journal_id, paper_id, title, annotation, citation, vector_id)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (journal_id, paper_id, title, annotation, citation, vector_id))
        conn.commit()
        print(f"Paper {journal_id}-{paper_id} added")
    except sqlite3.IntegrityError:
        print("Paper already exists")
        pass
    conn.close()

# adds paper to experimental table
def add_experiment_paper(link, abstract):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print('Connected to db')

    cursor.execute("""
            SELECT 1 FROM experimental WHERE link_to_paper = ?
        """, (link,))
    result = cursor.fetchone()

    if result is not None:
        conn.close()
        print("paper with that link exists")
        return

    print('paper is unique')

    next_faiss_id = faiss_controller.num_of_vectors(faiss_controller.experimental_path)

    print( f'next faiss id {next_faiss_id}')

    cursor.execute('''SELECT 1 FROM experimental WHERE faiss_id = ?
        ''', (next_faiss_id,))
    result = cursor.fetchone()

    if result is not None:
        conn.close()
        print("paper with that faiss id exists")
        return

    cursor.execute("""
    INSERT INTO experimental (faiss_id, link_to_paper, abstract)
    VALUES (?, ?, ?)
    """, (next_faiss_id, link, abstract))

    print('db entry added')

    embedding = gai_controller.get_embedding(abstract)
    faiss_controller.add_to_faiss(embedding, faiss_controller.experimental_path)

    conn.commit()
    conn.close()

    print('sqlite saved')

# gets id from journals table
def get_journal_id(name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM journals WHERE name = ?", (name,))
    journal = cursor.fetchone()
    conn.close()
    if journal:
        return journal[0]
    else:
        return None

# displays all rows of any table
def display_table_content(name, start=0, stop=50):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get total number of entries
    cursor.execute(f"SELECT COUNT(*) FROM {name}")
    total_entries = cursor.fetchone()[0]
    print(f"Total entries in '{name}': {total_entries}")

    # Fetch column names
    cursor.execute(f"PRAGMA table_info({name})")
    columns = [info[1] for info in cursor.fetchall()]
    print(" | ".join(columns))  # Display column names

    # Fetch rows within the specified range
    cursor.execute(f"SELECT * FROM {name} LIMIT ? OFFSET ?", (stop - start, start))
    rows = cursor.fetchall()

    for row in rows:
        print(row)

    conn.close()

# was used to move data from json to sqlite
def transfer_papers_from_json():
    json_path = '../data/json/math.json'
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    i = 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for paper in data["papers"]:
        try:
            cursor.execute('''
                    INSERT INTO papers (journal_id, paper_id, title, annotation, citation, vector_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                1,  # journal_id (assumed static in this example)
                paper['id'],
                paper['title'],
                paper['abstract'],
                paper['citation'],
                i  # vector_id
            ))
            conn.commit()
            print(f"Paper added: https://bm.kaznu.kz/index.php/kaznu/article/view/{paper['id']} ")
        except sqlite3.IntegrityError:
            print("Paper already exists")
        i+=1

    conn.close()

# papers table
def get_paper_by_embedding_id(vector_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT annotation FROM papers WHERE vector_id = ?", (vector_id,))
    abstract = cursor.fetchone()[0]
    conn.close()
    return abstract


# journals table
def check_journal_by_name(name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    id = get_journal_id(name)

    if id:
        return id
    else:
        cursor.execute("""
                INSERT INTO journals (name) VALUES (?)
            """, (name,))
        id = cursor.lastrowid
        conn.commit()

    conn.close()
    return id

# papers and journals tables
def paper_exists(journal_id, paper_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 1 FROM papers WHERE journal_id = ? AND paper_id = ?
    """, (journal_id, paper_id))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


if __name__ == "__main__":
    # add_journal('https://bm.kaznu.kz/index.php/kaznu/')
    #view_table('experimental')

    #add_experiment_paper('https://doi.org/10.1093/comjnl/7.2.149', '''A quadratically convergent gradient method for locating an unconstrained local minimum of a function of several variables is described. Particular advantages are its simplicity and its modest demands on storage, space for only three vectors being required. An ALGOL procedure is presented, and the paper includes a discussion of results obtained by its used on various test functions.''')


    display_table_content('experimental')
    #fetch_all_tables()
