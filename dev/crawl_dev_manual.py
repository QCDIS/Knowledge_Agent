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
    # FOR CODE, USE ONLY DESCRIPTION EMBEDDING
    #embedding = await get_embedding(extracted['title'] + "\n" + extracted['summary'] + "\n" + chunk)
    embedding = await get_embedding(extracted['title'] + "\n" + extracted['summary'])
    # Create metadata
    metadata = {
        "source": "environmental_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    # urlparse(url).path
    # " ".join(url)
    

    # " ".join(url),
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
    print("Printing chunks ...", chunks)
    
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
        urls = ["https://zenodo.org/communities/envrihubnext/records?q=&l=list&p=1&s=10&sort=newest", 
                "https://zenodo.org/communities/envrihubnext/records?q=&l=list&p=2&s=10&sort=newest"]
        

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
   
    # urls = get_pydentic_ai_docs_urls_once()
    
  
    # # CRAWL ALL (UNCOMMENT THIS WHEN TESTING DONE)
    # if not urls:
    #     print("No URLs found to crawl")
    #     return
    
    # print(f"Found {len(urls)} URLs to crawl")
    # await crawl_parallel(urls)
    url = "https://eosc.eu/wp-content/uploads/2024/05/EOSC-A_GA8_20240527-28_Paper-G_Update_EOSC_Nodes_requirements-DRAFT-v240524.pdf"
    text = """

    Building the EOSC Federation:requirements for EOSC Nodes[draft v 24/5]Disclaimer: This document is primarily addressing entities interested in enrolling as EOSC Nodes.The requirements will evolve during the uptake of the EOSC Federation and are planned to bespecified in the EOSC Federation Handbook. The document is not about the onboarding of serviceproviders to EOSC Nodes, and it does not elaborate on the steps to be taken for the uptake of theEOSC Federation.The European Open Science Cloud (EOSC), the Common European Data Space for R&I1, is theEU’s flagship initiative for the digital transformation of research.The vision for EOSC is to put in place a system for researchers in Europe to store, share, process,analyse and reuse, within and across disciplines and borders, FAIR research outputs, such asresearch data, publications and software.It is rooted in open science and deployed as a network between data repositories and services ofresearch infrastructures and other scientific service providers to develop a trusted federation ofresearch data and services for R&I in Europe. The development of the building blocks of the EOSCFederation will count on the latest community know-how, adopting and building on well-established and well-functioning existing structures and frameworks.EOSC is transitioning to a fully operational mode: the EC-procured ‘Managed Services for theEOSC Platform’ are currently being developed and gradually put in operation under the brandingof the EOSC EU Node2, making it the first node of the EOSC Federation.At the same time, partly through investments by the EU, its Member States and Horizon Europeassociated countries, several thematic communities and national initiatives are maturing theirinfrastructures and services for EOSC readiness.The EOSC Nodes will be the entry points for end users to the entire EOSC Federation, with eachnode offering its own services (including data reposing and accessing services) and possiblyservices of other providers (‘third-party’ services onboarded to the EOSC Node), in compliancewith the EOSC Federation’s common rules and requirements as well as the own policies of theEOSC Node. An organisation may therefore enrol its activities as an EOSC Node in the EOSCFederation or onboard its services on an existing EOSC Node as a provider.These nodes will thus offer (scientific) services and/or data that adhere to the FAIR principles,including curated research outputs (such as publications, datasets, software, etc.), specialised1 https://digital-strategy.ec.europa.eu/en/library/second-staff-working-document-data-spaces2 https://open-science-cloud.ec.europa.eu/2knowledge, applications, tools, infrastructure and/or platform services, and/or data processingand storage capabilities.Organisations that enrol their activities in the EOSC Federation as nodes will benefit from:● Broader outreach and connectivity: By enrolling in the EOSC Federation as nodes,organisations will gain access to a wider pool of users, within and especially beyond theirspecific thematic area and/or geographical focus, enabling innovative use and exploitationof their data and services, and potentially expanding beyond research to industry and thepublic sector as EOSC will progressively connect to the other currently deployed CommonEuropean Data Spaces. The end users will profit from a much broader access to FAIR dataand interoperable services through the EOSC Federation, enriching their own research, aswell as a widened recognition of their own contributions to the EOSC Federation.● Economies of scale: EOSC Nodes may take advantage of commonly pooled resources andcapabilities across the federation such as AAI, resource catalogues and registry services,monitoring, accounting, helpdesk etc, reducing duplication of research and developmentcosts by each node and strengthening their business models.● Europe-wide standards and policies: Organisations enrolling nodes in the EOSC Federationwill take part in shaping and adopting the latest European standards for research outputmanagement and service provision and will seamlessly comply with EU research and digitalpolicies as embodied in the EOSC Federation rules and policies.Below we set out the requirements for organisations to enrol as EOSC Nodes or to be considereda ‘candidate node’ already in the early uptake of the EOSC Federation.Requirements for EOSC NodesOrganisations responsible for EOSC Nodes shall ensure the quality of services offered by and tothe EOSC Federation, which translates into specific responsibilities towards other EOSC Nodes,towards third-party service providers and towards their end users.The requirements for an organisation that is responsible for an EOSC Node aim to ensure that itwill have the operational, administrative and legal capacity to take up this responsibility and to beable to contribute to the continuous development of EOSC according to its long-term vision.These requirements notably include:● Legal status: the organisation responsible for the EOSC Node must be a public-benefit legalentity (for now located in an EU Member State or an associated country) with legalpersonality and full legal capacity recognised in all Member States and associatedcountries, or an intergovernmental research organisation of European interest. Theorganisation must be able to conclude possible agreements with other partnersparticipating in the activities of the node itself (e.g. providers making their data or servicesavailable to EOSC through the node), with other nodes and/or with a potential futureorganisation representing the EOSC Federation.● Large-scale, quality service provision: EOSC Nodes shall be able to provide services atscale that are commonly used and endorsed by the research communities, operating in acompliant, sovereign, and secure environment.3● Capacity to onboard third-party services: beyond offering its own services, an EOSC Nodemay have the capacity to onboard third-party services on it and to ensure that these servicescomply with the common quality standards, rules, and policies of the federation, includingthose related to security, sovereignty, transparency, and trustworthiness of these services.● Capacity to contribute to EOSC core capabilities: EOSC Nodes shall have the capacity toutilise and contribute to specified core capabilities to be offered across the federation suchas Authentication and Authorization Infrastructure, resource catalogues and registryservices, monitoring, accounting and helpdesk.● Compliance with EOSC federation rules and standards: organisations that are responsiblefor EOSC Nodes retain autonomy to select which services they offer or share within theEOSC Federation and to set specific policies for access and use of these services, includingpricing-related policies for cost-intensive services. EOSC Nodes shall provide access totheir services under documented policies and be able to comply or to provide action plan toachieve such compliance with possible federation-wide agreed policies, protocols,standards and participation and access rules, including the EOSC interoperabilityframework, security (incl. cybersecurity) and sovereignty standards.● Effective monitoring: EOSC Nodes shall be able to monitor and report the activity of theservices they provide within the EOSC Federation (e.g. monitoring usage of data, servicesand other relevant activities) to ensure the quality of the provided services, including theonboarded services provided by third parties, and the compliance with the Federation’srules and standards.● Community engagement: EOSC Nodes will strive to contribute to community engagementactivities of the EOSC Federation, such as training activities, consultations, usability testing,communication, etc.● Sustainability: EOSC Nodes shall be able to confirm continued operations compliant to theEOSC Federation’s requirements for ideally 5 years or more to ensure that they are reliablemembers of the EOSC Federation, and they shall be transparent about their measures toguarantee the necessary lifetime.The above requirements set the framework and will be refined and elaborated in the months tocome, based also on initial experiences with candidate nodes. Estimates about costs and benefitswill be collected from the ‘candidate EOSC Nodes’.Minimal requirements for ‘candidate EOSC Nodes’It is foreseen to start with an initial critical mass of prospective EOSC Nodes, possibly includingnational nodes (nodes related to a certain country / nation), thematic nodes, (single sited and/ordistributed) research infrastructure nodes, or e-infrastructure nodes, and then to progressivelyexpand the federation according to the readiness of more prospective nodes and to evolve therequirements accordingly, based on the experience gained by the initial operation of the EOSCFederation.The minimum requirements for candidate EOSC Nodes are a subset of the requirements for EOSCNodes, in particular regarding the legal status, sustainability, core capabilities/service provisionas well as rules and standards:4● The legal entity representing the candidate EOSC Node must have the authority to establishagreements.● The legal entity responsible for the candidate EOSC Node should commit to providingsufficient resources to ensure its operation for at least 24 months, ideally for 5 years ormore.● The candidate EOSC Node must have sufficient capacity and expertise to be able to ensurethat the data and services it includes can be operated at a sufficiently high Technology andReadiness Level ensuring robust, reliable and secure performance.● Data and Services must be at least findable and accessible to registered EOSC users andfulfil the FAIR principles in general, while services such as compute and storage capacityshould be available in sufficient quantities to multiple user groups.    


    """
    await process_and_store_document(url, text)
    

if __name__ == "__main__":
    asyncio.run(main())



# How to read dataset for EXV and temporal coverage

# ENVRI Hub Library usage for Catalogue Search

# Build a client code for the ANAEE weather API using a specification file

# How to find DAO objects for auto-generated resource metadata?
    # show sample code (N.B. This question can be done as a follow up with other questions)

# How to install envrihub library and use it?
# How to access a resource directly with unique identifier in the Catalogue of Services?

# How to get help with DAO objects for ENVRI?

# How to access the envri-hub services through programming ?

# How to access RI data and services using the envri hub ? (here I would see the CoS and the envrihub library/example notebook).



