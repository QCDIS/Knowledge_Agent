import asyncio
import random
import pandas as pd
import markdown
from bs4 import BeautifulSoup
from typing import List, Dict
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
from openai import AsyncOpenAI
from supabase import create_client, Client

# Load environment variables
load_dotenv()
#     "name": ["Data set name", "Data holding centre", "Originators", "Organisation", "Collating centre"],
#     "description": ["Parameters", "Instruments", "Summary", "Availability"],

CATEGORY_MAPPING = {
    "name": ["Data holding centre", "Originators", "Organisation", "Collating centre"],
    "place": ["Country", "Geographical area"],
    "time": ["Time period", "Last revised"],
    "description": ["Parameters", "Instruments", "Availability"],
    "other_numeric": ["Local identifier", "Global identifier", "Telephone"]
}

async def categorize_key(key: str) -> str:
    """Categorize a given key into one of the predefined categories."""
    key_lower = key.lower()
    for category, keywords in CATEGORY_MAPPING.items():
        if any(keyword.lower() in key_lower for keyword in keywords):
            return category
    return None  # Return None if no match is found

file_records = {}  # Stores extracted values per file for random pairing
extracted_data = []


async def process_markdown(markdown_text: str, url: str):
    """Process and parse markdown content, extract structured data, and save results."""
    print("Processing markdown...----------------------------------------------------")

    # Convert Markdown to HTML
    html_content = markdown.markdown(markdown_text)

    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract structured information
    parsed_data = {}
    sections = ["General", "Observations", "Description", "Availability", "Administration"]
    current_section = None

    for line in soup.stripped_strings:
        if line in sections:
            current_section = line
            parsed_data[current_section] = {}
        elif current_section:
            if "|" in line:  # Key-Value pairs from Markdown table format
                key, value = map(str.strip, line.split("|", 1))
                parsed_data[current_section][key] = value

    print("Parsed Data:", parsed_data)


    file_name = f"{url}"
    file_records[file_name] = []
    for section, values in parsed_data.items():
        print("Inside loop ..", values)
        # Inside loop .. {'Organisation': 'Total Oil Marine Plc', 'Availability': 'By negotiation', 'Contact': 'The Director', 'Address': 'Total Oil Marine Plc Crawpeel Road Altens Aberdeen AB12 3AG United Kingdom', 'Telephone': '+44 1224 858000'}
        if isinstance(values, dict):  # Ensure it's a dictionary
            for key, value in values.items():
                category = await categorize_key(key)
                if category:
                    extracted_entry = {
                        "File": file_name,
                        "Category": category,
                        "Key": key,
                        "Value": str(value)  # Convert all values to string for uniformity
                    }
                    print("Printing one extracted category ...", extracted_entry)

                    extracted_data.append(extracted_entry)
                    file_records[file_name].append(str(value))

    # Generate random pairs
    random_pairs = []
    file_names = list(file_records.keys())
    print("Printing file names ...", file_names)
    if len(file_names) > 1:
        print("inside 2nd if ...")
        for file1 in file_names:
            if not file_records[file1]:
                continue  # Skip empty files
            
            # Select a random value from file1
            value1 = random.choice(file_records[file1])
            
            # Find another file to pair with
            file2 = random.choice([f for f in file_names if f != file1 and file_records[f]])
            value2 = random.choice(file_records[file2])
            
            # Store the paired values
            random_pairs.append({
                "Pair 1 (Value)": value1,
                "Pair 2 (Value)": value2,
                "Pair 1 File Name": file1,
                "Pair 2 File Name": file2
            })

    # Save extracted values
    df_extracted = pd.DataFrame(extracted_data)
    df_extracted.to_csv("extracted_values_test.csv", index=False)
    print("Extracted values saved to extracted_values_test.csv")

    # Save random value pairs
    df_pairs = pd.DataFrame(random_pairs)
    df_pairs.to_csv("random_value_pairs_test.csv", index=False)
    print("Random value pairs saved to random_value_pairs_test.csv")

async def process_url(crawler: AsyncWebCrawler, url: str, semaphore: asyncio.Semaphore, crawl_config: CrawlerRunConfig):
    """Process a single URL using the crawler with concurrency control."""
    async with semaphore:
        try:
            result = await crawler.arun(url=url, config=crawl_config, session_id="session1")
            if result.success:
                print(f"Successfully crawled: {url}")
                await process_markdown(result.markdown.raw_markdown, url)
            else:
                print(f"Failed: {url} - Error: {result.error_message}")
        except Exception as e:
            print(f"Error processing {url}: {e}")

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
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
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [process_url(crawler, url, semaphore, crawl_config) for url in urls]
        await asyncio.gather(*tasks)
    finally:
        await crawler.close()

def get_seadatanet_urls(start_id: int = 850, end_id: int = 900) -> List[str]:
    """Generate a list of URLs to crawl from the SeaDataNet EDMED directory."""
    return [f"https://edmed.seadatanet.org/report/{i}/" for i in range(start_id, end_id)]

async def main():
    """Main async function to fetch URLs and start crawling."""
    urls = get_seadatanet_urls()
    if not urls:
        print("No URLs found to crawl")
        return

    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)


#################################

single_coupling_list = []
def single_coupling(fileName):

    df = pd.read_csv(fileName)
    print(df.head(5))
    
    for index_1, rows_1 in df.iterrows():
        match = False
        url_1 = rows_1["File"]
        value_1 = rows_1["Value"]

        #print(index_1)

        for index_2, rows_2 in df.iloc[index_1 + 1:].iterrows():

            value_2 = rows_2["Value"]
            if value_1 == value_2:
                #print("Match Found", value_2)
                url_2 = rows_1["File"]
                #print(index_1, index_2)

                extracted_entry = {
                    "URL_1": url_1,
                    "URL_2": url_2,
                    "Value": str(value_2)  # Convert all values to string for uniformity
                }
                single_coupling_list.append(extracted_entry)
                match = True
        if match == False:
            extracted_entry = {
                "URL_1":url_1,
                "URL_2":url_1,
                "Value": str(value_1)
            }
    
    print(single_coupling_list[0])
    df_single_coupling_list = pd.DataFrame(single_coupling_list)


    df_merged = df_single_coupling_list.groupby("Value").agg({
        "URL_1": lambda x: "; ".join(set(x)),  
        "URL_2": lambda x: "; ".join(set(x))   
    }).reset_index()

    # Combine URL_1 and URL_2 into a single "URL" column
    df_merged["URL"] = df_merged["URL_1"] + "; " + df_merged["URL_2"]

    # Drop the old columns
    df_merged = df_merged.drop(columns=["URL_1", "URL_2"])

    # Display the final DataFrame
    print(df_merged)
    df_merged.to_csv("df_single_coupling_list.csv")


if __name__ == "__main__":
    asyncio.run(main())
    fileName = "extracted_values_test.csv"
    single_coupling(fileName=fileName)
