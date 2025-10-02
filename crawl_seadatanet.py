# development started

import json
import re
from xml.etree import ElementTree
import urllib.request
import string


import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import textwrap
import abc


from tqdm import tqdm
import urllib.request
import urllib.error
import requests

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()





openai_api_key = os.getenv("OPENAI_API_KEY")
db_password = os.getenv("DB_PASSWORD")
supabase_url = os.getenv("SUPABASE_URL")
supabase_secret = os.getenv("SUPABASE_SECRET")

openai_client = AsyncOpenAI(api_key=openai_api_key)
supabase: Client = create_client(
    supabase_url,
    supabase_secret
)


def sparql_query(endpoint, query, max_records=None, offset=0):
        if max_records is not None:
            query += f'limit {max_records}\n'
        if offset:
            query += f'offset {offset}\n'
        query = textwrap.dedent(query).strip()

        r = requests.post(
            endpoint,
            headers={
                'Cache-Control': 'no-cache',
                'accept': 'text/csv'},
            data={'query': query},
            )
        return r.text.splitlines()[1:]
    

def getUrlsVOC(max_records=None, offset=0):
    documents_list_url = 'https://vocab.nerc.ac.uk/sparql'
    document_extension = 'html'
    query = r"""
        SELECT ?o WHERE {
        ?s ?p ?o .
        FILTER(STRSTARTS(STR(?o), "http://vocab.nerc.ac.uk/collection/"))
    }
    """
    urls = sparql_query(
        documents_list_url,
        query,
        max_records=max_records,
        offset=offset,
        )[:-1]
    
    #print("Printing URLS ...", urls)
    
    return [f'{url}' for url in urls]

def getUrlsEDMED(max_records=None, offset=0):
    documents_list_url = 'https://edmed.seadatanet.org/sparql/sparql'
    document_extension = '.html'       
    query = r"""
    select ?EDMEDRecord
    where {
        ?EDMEDRecord a <http://www.w3.org/ns/dcat#Dataset>
    }
    """
    return sparql_query(
        documents_list_url,
        query,
        max_records=max_records,
        offset=offset,
        )[:-1]
    



# seadatadanet vocabulary https://vocab.nerc.ac.uk/sparql/
def getUrlsCDI(max_records=None, offset=0):
    documents_list_url = 'https://cdi.seadatanet.org/sparql/sparql'
    document_extension = '.json'
    query = r"""
    SELECT ?o WHERE {
        ?s <http://www.w3.org/ns/dcat#dataset> ?o
    }
    """
    urls = sparql_query(
        documents_list_url,
        query,
        max_records=max_records,
        offset=offset,
        )[:-1]
    return [f'{url}/json' for url in urls]


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)
    print("Printing text ...", text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

        #print("Printing Chunk Text", chunk)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    #print("Printing chunks ...", chunks)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    #Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    #crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.6),
            options={"ignore_links": True}
            
        ),
        cache_mode=CacheMode.BYPASS
    )
    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()



def get_pydantic_ai_docs_urls():
    """
    Fetches all URLs from the Pydantic AI documentation.
    Uses the sitemap (https://ai.pydantic.dev/sitemap.xml) to get these URLs.
    
    Returns:
        List[str]: List of URLs
    """            
    # urls = getUrlsVOC(100, 0)
    # print(urls)

    # urls1 = getUrlsCDI(100, 0)
    # print(urls)
    # urls2 = getUrlsEDMED(100, 0)
    urls = getUrlsVOC(10, 0) + getUrlsCDI(10, 0) + getUrlsEDMED(10, 0)

    print(urls)
    
    
    return urls

async def processIAGOS(folder):
    

    json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
    print(json_files)
    for file in json_files:
        with open(folder + file) as f:
            data = json.load(f)
            print(data["Description"])

            chunk_content = " Provider: " + " ".join(data["Provider"]) + " " + " ".join(data["Description"]) + " " + "Callable API Endpoint: " + " ".join(data["Endpoint Url"]) + " " + " Category: " + " ".join(data["Category"]) + " " + " Architectural Style: " + " " + " ".join(data["Architectural Style"]) + " Support SSL: " + " " + " ".join(data["Support SSL"])

            extracted = await get_title_and_summary(chunk_content, " ".join(data["Url"]))
    
            # Get embedding
            
            # Create metadata
            metadata = {
                "source": "pydantic_ai_docs",
                "chunk_size": len(chunk_content),
                "crawled_at": datetime.now(timezone.utc).isoformat(),
                "url_path": urlparse(" ".join(data["Endpoint Url"])).path
    }
         
            
            embedding = await get_embedding(chunk_content)
            chunk = ProcessedChunk(
                url = " ".join(data['Url']),
                chunk_number = 0,
                title=extracted['title'],
                summary=extracted['summary'],
                content = chunk_content,
                metadata= metadata,
                embedding=embedding
            )

            await insert_chunk(chunk)



async def main():
    # IAGOS_folder = "/home/nafis/Development/Knowledge_Agent/RI_Data/IAGOS/"
    # await processIAGOS(IAGOS_folder)

    #return 
    # Get URLs from Pydantic AI docs
    urls = get_pydantic_ai_docs_urls()
    return
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())



# urls = getUrlsVOC()
# print(urls)