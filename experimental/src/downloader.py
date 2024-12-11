import os
import requests

def read_pubmed_ids(file_path):
    """Read PubMed IDs from a file."""
    with open(file_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids

def fetch_abstract(pubmed_id):
    """Fetch the abstract text for a given PubMed ID."""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pubmed_id,
        "rettype": "abstract",
        "retmode": "text"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch abstract for PubMed ID {pubmed_id}. Status code: {response.status_code}")
        return None

def save_abstract(text, pubmed_id, save_dir):
    """Save the abstract text to a file."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{pubmed_id}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved abstract for PubMed ID {pubmed_id} to {file_path}")

def main():
    # HARDCODED CHANGE THESE INPUT OUTPUT PATHS:
    
    # Path to the file containing PubMed IDs
    ids_file = os.path.join(os.path.dirname(__file__), '../resources/sample_abstracts_ids.txt')
    # Directory where abstracts will be saved
    save_directory = os.path.join(os.path.dirname(__file__), '../resources/sample_abstracts')

    pubmed_ids = read_pubmed_ids(ids_file)
    print(f"Found {len(pubmed_ids)} PubMed IDs.")

    for pubmed_id in pubmed_ids:
        abstract_text = fetch_abstract(pubmed_id)
        if abstract_text:
            save_abstract(abstract_text, pubmed_id, save_directory)

if __name__ == "__main__":
    main()
