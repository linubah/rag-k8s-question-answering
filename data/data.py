import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import pandas as pd
import os

docs_csv = 'data/docs.csv'
dataset_columns = ["chunk_id", "content", "title", "source", "document_id", "order"]

df = pd.read_csv('data/data_links.csv')
links = df['link'].tolist()


def get_page(url: str) -> list:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = soup.find('div', class_='td-content')
        h1_tag = main_content.find('h1')
        h1_text = h1_tag.get_text(strip=True)
        return main_content, h1_text
    except requests.RequestException as e:
        return "Error: " + str(e)

def extract_chunks(content: Tag) -> list:
    chunks = []
    current_chunk = []

    for tag in content.find_all(['h2', 'h1', 'p', 'ul', 'ol', 'li', 'div']):
        if tag.name == 'h2' or tag.name == 'h1':
            if current_chunk: 
                chunks.append("\n".join(current_chunk))
            current_chunk = [tag.get_text()] 
        else:
            current_chunk.append(tag.get_text()) 
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def get_last_record(file_path: str, columns: list) -> int:
    if os.path.exists(file_path):
        docs_df = pd.read_csv(file_path)
        
        if not docs_df.empty and "chunk_id" in docs_df.columns:
            last_chunk_id = docs_df["chunk_id"].iloc[-1]
            print(f"Last chunk_id: {last_chunk_id}")
        else:
            last_chunk_id = 0
    else:
        last_chunk_id = 0
        docs_df = pd.DataFrame(columns=columns)
        docs_df.to_csv(file_path, index=False)

    return last_chunk_id, docs_df


def generate_dataset(links: list):
    last_chunk_id, docs_df = get_last_record(docs_csv, dataset_columns)
    documents = []

    for url in links:
        url = url.strip("/")
        page, title = get_page(url)
        if not page:
            print("Error: page not found")
            continue

        chunks = extract_chunks(page)

        for i, chunk in enumerate(chunks, 1):
            last_chunk_id += 1
            document = {
                "chunk_id": last_chunk_id,
                "content": chunk,
                "title": title,
                "source": url,
                "document_id": "/".join(url.split('/')[-2:]),
                "order": i
        }
            documents.append(document)

                
    new_rows = pd.DataFrame(documents)
    docs_df = pd.concat([docs_df, new_rows], ignore_index=True)
    docs_df.to_csv(docs_csv, index=False)

    print("Data saved successfully!")
    print("Total documents:", len(docs_df["document_id"].unique()))
    print("Total chunks:", len(documents))
