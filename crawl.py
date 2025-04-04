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

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
opai_api_key = "<OPENAI API KEY"
db_password = "<SUPABASE PASSWORD>"
supabase_url = "https://woqcpiqkhfhhmenrphph.supabase.co"
supabase_secret = "<SUPABASE SECRET>"
openai_client = AsyncOpenAI(api_key=opai_api_key)
supabase: Client = create_client(
    supabase_url,
    supabase_secret
)

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

        print("Printing Chunk Text", chunk)

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
    try:
        urls = []
        for i in range(850, 900):
            urls.append("https://edmed.seadatanet.org/report/" + str(i) + "/")
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []       


async def main():
    # Get URLs from Pydantic AI docs
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())

"""

[![Back to home](https://edmed.seadatanet.org/images/seadatanet_logo_big.png)](https://edmed.seadatanet.org/report/854/<http:/www.seadatanet.org>)
Pan-European infrastructure for ocean & marine data management
# European Directory of Marine Environmental Data (EDMED)
## Data set information
| [Query EDMED](https://edmed.seadatanet.org/report/854/</search/> "Query EDMED — access key 'Q'") | 
**General**  
---  
Data set name| 
# Wind and wave data from North Sea Platforms (1974-1987)  
Data holding centre| [United Kingdom Offshore Operators Association](https://edmed.seadatanet.org/report/854/</org/83/> "View data centre information — access key 'D'")  
Country| United Kingdom ![United Kingdom](https://edmed.seadatanet.org/images/flags/gb.gif)  
Time period| Various periods between 1974 and 1987  
Ongoing| No  
Geographical area| North Sea  
**Observations**  
Parameters| Wind strength and direction; Wave direction; Spectral wave data parameters; Wave height and period statistics  
Instruments| Anemometers; wave recorders  
**Description**  
Summary| The data set comprises various measurements of winds and waves, mostly collected by Marex (now Paras), on behalf of UKOOA. Wind data from Brent Platform and wind and wave data from North Cormorant were gathered by Shell. The data set is detailed below. Site Latitude Longitude Water Start Date End Date (° min) (° min) Depth (m) Beryl Field 59 30N 001 30E 140 01 Jan 1979 31 Oct 1981 Foula Data Buoy 60 07.5N 002 57W 155 05 Dec 1976 31 Dec 1978 Brent Platform 61 04N 001 43E 140 01 Jan 1978 31 May 1980 North Cormorant 61 14N 001 10E 161 01 Sep 1983 31 Aug 1987 Forties FB 57 40N 000 50E 128 01 Dec 1976 31 May 1980 Forties Field 57 40N 000 50E 128 20 Jun 1974 31 May 1977 Note that at Beryl Field wind data collection began on 01 October 1976. Also, at Forties Field, wind data are a composite from various vessels between 20 June 1974 and 21 October 1975 and were compiled from Forties FC, FA and Montrose A between 29 November 1975 and 30 November 1976. Most of the above data, except that at North Cormorant, are also held by the British Oceanographic Data Centre (BODC).  
Originators| [United Kingdom Offshore Operators Association](https://edmed.seadatanet.org/report/854/</org/83/>)  
**Availability**  
Organisation| [United Kingdom Offshore Operators Association](https://edmed.seadatanet.org/report/854/</org/83/> "View organisation information — access key 'D'")  
Availability| By negotiation  
Contact| The Director  
Address| United Kingdom Offshore Operators Association 3 Hans Crescent London SW1X 0LN United Kingdom  
Telephone| +44 171 589 5255  
**Administration**  
Collating centre| [British Oceanographic Data Centre](https://edmed.seadatanet.org/report/854/</org/43/> "View collating centre information — access key 'C'")  
Local identifier| 1089002  
Global identifier| 854  
Last revised| 2009-10-15  
EDMED service is provided by the British Oceanographic Data Centre ©2025
Page dynamically generated: March 01, 2025

"""