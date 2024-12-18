import os
import json
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
from Bio import Entrez
from Bio import Medline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_pubmed_articles(
    start_date: str,
    end_date: str,
    max_articles: int = 100,
    email: str = "",
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    Fetches PubMed articles within a date range.

    Args:
        start_date (str): Start date in format 'YYYY/MM/DD'.
        end_date (str): End date in format 'YYYY/MM/DD'.
        max_articles (int): Maximum number of articles to fetch.
        email (str): User email address (required by NCBI).
        api_key (str, optional): NCBI API key to increase rate limits.

    Returns:
        List[Dict]: A list of articles with their metadata.
    """
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    # Build the search term for date range
    search_term = f'"{start_date}"[Date - Publication] : "{end_date}"[Date - Publication]'

    logger.info(f"Searching PubMed for articles from {start_date} to {end_date}")
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_articles)
    record = Entrez.read(handle)
    handle.close()
    id_list = record["IdList"]
    total_count = int(record["Count"])
    logger.info(f"Found {total_count} articles in the date range")

    # Adjust max_articles if fewer articles are available
    max_articles = min(max_articles, total_count)
    logger.info(f"Fetching up to {max_articles} articles")

    articles = []
    batch_size = 100  # NCBI recommends fetching records in batches
    for start in range(0, max_articles, batch_size):
        end = min(max_articles, start + batch_size)
        batch_ids = id_list[start:end]
        logger.info(f"Fetching records {start + 1} to {end}")
        fetch_handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="text")
        records = Medline.parse(fetch_handle)
        for record in records:
            article = parse_medline_record(record)
            if article:
                articles.append(article)
        fetch_handle.close()
        time.sleep(0.5)  # To respect NCBI rate limits

    logger.info(f"Fetched {len(articles)} articles with abstracts")
    return articles

def parse_medline_record(record) -> Optional[Dict]:
    """
    Parses a Medline record into the desired format.

    Args:
        record: A Medline record.

    Returns:
        Dict or None: Parsed article data or None if abstract is missing.
    """
    pmid = record.get('PMID')
    title = record.get('TI') or record.get('Title')
    abstract = record.get('AB') or record.get('Abstract')
    authors = record.get('AU') or []
    journal = record.get('JT') or record.get('TA') or record.get('Journal')
    year = None
    if 'DP' in record:
        year_match = record['DP'][:4]
        if year_match.isdigit():
            year = int(year_match)

    if not abstract:
        logger.warning(f"No abstract found for PMID {pmid}, skipping")
        return None

    article = {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "year": year
    }
    return article

if __name__ == "__main__":
    # User inputs
    print("Enter date range for PubMed articles.")
    date_option = input("Type '1' for last 2 years, '2' for last 4 years, or '3' to specify custom dates: ").strip()

    if date_option == '1':
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)
    elif date_option == '2':
        end_date = datetime.now()
        start_date = end_date - timedelta(days=4*365)
    elif date_option == '3':
        start_date_input = input("Enter start date (YYYY/MM/DD): ").strip()
        end_date_input = input("Enter end date (YYYY/MM/DD): ").strip()
        try:
            start_date = datetime.strptime(start_date_input, '%Y/%m/%d')
            end_date = datetime.strptime(end_date_input, '%Y/%m/%d')
        except ValueError:
            logger.error("Invalid date format. Please use YYYY/MM/DD.")
            exit(1)
    else:
        logger.error("Invalid option selected.")
        exit(1)

    max_articles = int(input("Enter maximum number of articles to fetch: ").strip())
    email = ''
    api_key = ''

    # Convert dates to strings in required format
    start_date_str = start_date.strftime('%Y/%m/%d')
    end_date_str = end_date.strftime('%Y/%m/%d')

    # Fetch articles
    articles = fetch_pubmed_articles(
        start_date=start_date_str,
        end_date=end_date_str,
        max_articles=max_articles,
        email=email,
        api_key=api_key
    )

    # Save articles to data.json
    if articles:
        os.makedirs("data", exist_ok=True)
        output_file = os.path.join("data", "data.json")
        with open(output_file, 'w') as f:
            json.dump(articles, f, indent=4)
        logger.info(f"Saved {len(articles)} articles to {output_file}")
    else:
        logger.info("No articles were fetched.")
