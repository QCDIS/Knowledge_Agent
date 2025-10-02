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



@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)
    print("Printing text ...", type(text))

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
    DELAY_MS = 140  # <- 110 milliseconds

    try:
        await asyncio.sleep(DELAY_MS / 1000)
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
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
        # text-embedding-3-large
        # text-embedding-3-small
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
    embedding = await get_embedding(extracted['title'] + "\n" + extracted['summary'] + "\n" + chunk)
    
    # Create metadata
    metadata = {
        "source": "environmental_docs",
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
        
        result = supabase.table("site_pages_dev").insert(data).execute()
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

# one time use functions
def get_pydentic_ai_docs_urls_once():
    try:
        urls = ["https://github.com/nvs-vocabs/EXV", 
                "https://vocab.nerc.ac.uk/collection/EXV/current/", 
                "https://vocab.nerc.ac.uk/search_nvs/EXV/", 
                "https://library.wmo.int/records/item/58111-the-2022-gcos-ecvs-requirements"]

        return urls

    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

def get_pydantic_ai_docs_urls():
    """
    Fetches all URLs from the Pydantic AI documentation.
    Uses the sitemap (https://ai.pydantic.dev/sitemap.xml) to get these URLs.
    
    Returns:
        List[str]: List of URLs
    """  


    
    ## WORKING CODE FOR SEADATANET COMMENTED ###
    try:

        

        # sitemap_url_envri = ["https://envri.eu/wp-sitemap-posts-post-1.xml",
        # "https://envri.eu/wp-sitemap-posts-page-1.xml",
        # "https://envri.eu/wp-sitemap-taxonomies-category-1.xml",
        # "https://envri.eu/wp-sitemap-taxonomies-post_tag-1.xml"
        # ]
        # envri_urls = []
        # for sitemap in sitemap_url_envri:
        #     response = requests.get(sitemap)
        #     response.raise_for_status()
            
        #     root = ElementTree.fromstring(response.content)
            
        #     # Extract all URLs from the sitemap
        #     namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        #     envri_urls += [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        
        


        # seadatanet_urls = []
        # for i in range(850, 1000):
        #     seadatanet_urls.append("https://edmed.seadatanet.org/report/" + str(i) + "/")
       

        # icos_urls = ["https://meta.icos-cp.eu/objects/DLlJ-m63gJHSmiUipESYGuLd", 
        #         "https://meta.icos-cp.eu/objects/1IxrtKdX-IhHJdTM9UMrSsbj", 
        #         "https://meta.icos-cp.eu/objects/Mfe9bdiDJuMVXW7tIBJuJXMc",
        #         "https://meta.icos-cp.eu/objects/A5Fby8ZbD2I-enYvFDsQvNAE",
        #         "https://meta.icos-cp.eu/objects/2URn417sWYN0ROUJMIXjCiTq",
        #         "https://meta.icos-cp.eu/objects/O5lzanad8Yr_t0Doc8U1W4zS", 
        #         "https://meta.icos-cp.eu/objects/_1cSJaHszlKSAVAYBbp3VKzw", 
        #         "https://meta.icos-cp.eu/objects/yUXwLrwO2uRgSiKEorv-ZZfw", 
        #         "https://meta.icos-cp.eu/objects/yUXwLrwO2uRgSiKEorv-ZZfw",
        #         "https://meta.icos-cp.eu/objects/5DkSoXP7Y1SZV4SeO1nyLBQW", 
        #         "https://iagos.aeris-data.fr/download-instructions/", 
        #         "https://www.iagos.org/", 
        #         "http://www.argodatamgt.org/Access-to-data/Argo-GDAC-synchronization-service",
        #         "https://data.marine.copernicus.eu/product/INSITU_GLO_PHY_TS_DISCRETE_MY_013_001/description",
        #         "https://www.data-terra.org/pole-odatis/",
        #         "https://data.blue-cloud.org/search/argo",
        #         "https://www.data-terra.org/pole-odatis/",
        #         "https://envrihub.vm.fedcloud.eu/",
        #         "https://vocab.nerc.ac.uk/",
        #         "https://fleetmonitoring.euro-argo.eu/dashboard?Status=Active",
        #         "https://dataselection.euro-argo.eu/",
        #         "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.html",
        #         "https://stac-browser.ifremer.fr/?.language=en"]

        # with open("RI_Data/eLTER/urls.txt", "r") as file:
        #     elter_urls = [line.strip() for line in file if line.strip()]
        
        # with open("RI_Data/AnaEE/urls.txt", "r") as file:
        #     anaee_urls = [line.strip() for line in file if line.strip()]

        with open("RI_Data/ECV/urls.txt", "r") as file:
            ecv_urls = [line.strip() for line in file if line.strip()]

        # urls = seadatanet_urls + icos_urls + elter_urls + anaee_urls + ecv_urls + envri_urls
        with open("RI_Data/RI/explored_urls.txt", "r") as file:
            RI_urls = [line.strip() for line in file if line.strip()]
        
        urls = RI_urls + ecv_urls
        print("Printing URLs ", urls)

    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []    

    print("Total number of URLs ", len(urls))
    # https://gcos.wmo.int/site/global-climate-observing-system-gcos/essential-climate-variables
    # and GOOS page: https://goosocean.org/what-we-do/framework/essential-ocean-variables/. 
    # As you could see in this page: https://vocab.nerc.ac.uk/collection/EXV/current/, these 
    # urls = ["https://gcos.wmo.int/site/global-climate-observing-system-gcos/essential-climate-variables/clouds", 
    #         "https://gitlab.a.incd.pt/envri-hub-next/analytical-workflow-templates/-/blob/main/ICOS_data_access.ipynb?ref_type=heads"]
    return urls

async def processIAGOS(folder):
    

    json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
    print(json_files)
    for file in json_files:
        with open(folder + file) as f:
            data = json.load(f)
            print(data["Description"])

            chunk_content = " Provider: " + " ".join(data["Provider"]) + " " + " ".join(data["Description"]) + " " + "Callable API Endpoint: " + " ".join(data["Endpoint Url"]) + " " + " Category: " + " ".join(data["Category"]) + " " + " Architectural Style: " + " " + " ".join(data["Architectural Style"]) + " Support SSL: " + " " + " ".join(data["Support SSL"])
            
            
            #chunk_content = str(data["Description"]) + "Category: " + str(data["Category"]) + "Provider: " + str(data["Provider"]) + "Architectural Style:" + str(data["Architectural Style"]) + "Support SSL: " + str(data["Support SSL"])
            
            
            
            #print(chunk_content)
            
            #print(type(chunk_content))


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
    
    # CRAWL ALL (UNCOMMENT THIS WHEN TESTING DONE)
    #urls = get_pydantic_ai_docs_urls()



    # CRAWL ONCE
    ############ THIS IS MANUAL TASK FOR TESTING AND DEMO, MUST BE REMOVED ################
    urls = get_pydentic_ai_docs_urls_once()
    # with open("../RI_Data/code/envri_hub_library_usage.txt", "r") as file:
    #         code = " ".join([line.strip() for line in file if line.strip()])
    #         print(code)
    # manual_url = ["https://gitlab.a.incd.pt/envri-hub-next/analytical-workflow-templates/-/blob/main/ENVRI%20HUB%20library%20usage.ipynb?ref_type=heads"]
    # await process_and_store_document(manual_url, code)
    ##################
    
    
    # CRAWL ALL (UNCOMMENT THIS WHEN TESTING DONE)
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())

