from datasets import Dataset
import pandas as pd
import os
import asyncio
import numpy as np
import json
import itertools

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

"""
SAMPLE DATA FORMAT
data = {
    "prompt": ["What is AI?", "Another prompt"],
    "chosen": ["AI stands for Artificial Intelligence...", "Another Chosen"],
    "rejected": ["AI is a vegetable.", "Another response"],
}
"""
# https://huggingface.co/datasets/trl-internal-testing/descriptiveness-sentiment-trl-style

path = 'ratings/ratings/'

dir = os.listdir(path)
#print(dir)
#print(dfFinal)
def create_dataset_depricated(path):
    finalRatings = []
    # finalRatings = {"query": [],
    #                 "response": [],
    #                 "reward": []}
    
    finalRatings = {"prompt": [],
                    "chosen": [],
                    "rejected": []}
    
    files = os.listdir(path)
    #print(dir)
    counter = 0
    for file in files:
        df = pd.read_csv(path + file, index_col=False)
        
        dfLen = len(df)

        for index, rows in df.iterrows():

            finalRatings['query'].append(rows['Search Query'])
            reverseValue = dfLen - index - 1
            
            response = crawl(rows['Document ID'])
            finalRatings['chosen'].append(response)
            finalRatings['rejected'].append(rows['Rating'])
            counter += 1
            

        #break
        print("Value of counter ", counter)   
        if counter >= 10:
            break

    # REWARD ADD: Removed for now
    temp_npList = np.array(finalRatings['reward'], dtype=float)
    print("Original reward values ", temp_npList)
    if temp_npList.max() == temp_npList.min():
        normalized_npList = np.zeros_like(temp_npList, dtype=float)
    else:
        print("you are at else")
        normalized_npList = (temp_npList - 0) / (9 - temp_npList.min())
    print("Printing normalized reward values ", normalized_npList)
    finalRatings['reward'] = normalized_npList.tolist()

    print(len(finalRatings))
    json_ratings = json.dumps(finalRatings, indent=4)
    
    with open('test_ppo_data.json', 'w', encoding='utf-8') as f:
        json.dump(json_ratings, f, ensure_ascii=False, indent=4)

    return finalRatings
############################### Above method depricated for the time being ###############################################

def create_dataset(path):
    finalRatings = []
    
    
    finalRatings = {"prompt": [],
                    "chosen": [],
                    "rejected": []}
    
    files = os.listdir(path)
    #print(dir)
    counter = 0
    paired_rows = []
    for file in files:
        df = pd.read_csv(path + file, index_col=False)
        
        dfLen = len(df)

        # seperate 0-4 and 5-9
        # permute between them and add to finalRatings
        negative_responses = df[(df['Rating'] >= 0) & (df['Rating'] < 5)]
        positive_responses = df[(df['Rating'] >= 5) & (df['Rating'] <= 10)]
        print("NEG:" , negative_responses)
        print("POS:", positive_responses)
        
        
        # Create all combinations
        all_combinations = list(itertools.product(positive_responses.iterrows(), negative_responses.iterrows()))
        #print("ALL COMB", all_combinations)
        # Format combinations as a list of dicts, tuples, or DataFrame
        #print("Printing a row in a dataframe", df.loc[0, "Search Query"])

        for (pos_idx, pos_row), (neg_idx, neg_row) in all_combinations:
            paired_rows.append({
                "positive_index": pos_idx,
                "query" : df.loc[0, "Search Query"],
                "positive_doc_id": pos_row['Document ID'],
                "positive_rating": pos_row['Rating'],
                
                "negative_index": neg_idx,
                "negative_doc_id": neg_row['Document ID'],
                "negative_rating": neg_row['Rating'],
            })

            # Optional: convert to DataFrame
            
        #counter += 1
        print("--------------x--------------")
        #break
        print("Value of counter ", counter)   
        if counter >= 2:
            break
    
    
    print("len of paired rows ", len(paired_rows))
    pairs_df = pd.DataFrame(paired_rows)
    print(pairs_df.head())
    for index, rows in pairs_df.iterrows():

        finalRatings['prompt'].append(rows['query'])
        
        # a dict is appended to the list
        # USE CACHING FOR CRAWLING
        finalRatings['chosen'].append(crawl(pairs_df.loc[index, 'positive_doc_id']))
        finalRatings['rejected'].append(crawl(pairs_df.loc[index, 'negative_doc_id']))

        

    

    print(len(finalRatings))
    json_ratings = json.dumps(finalRatings, indent=4)
    
    with open('reward_data.json', 'w', encoding='utf-8') as f:
        json.dump(json_ratings, f, ensure_ascii=False, indent=4)

    return finalRatings


def crawl(url):
    async def runner():
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(url=url)

    result = asyncio.run(runner())
    #print(result.markdown)
    return result.markdown

def crawl(url):
    async def runner():
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(url=url)

    result = asyncio.run(runner())
    #print(result.markdown)
    return result.markdown
create_dataset(path)



##############################################################################
# Sample structure
"""
data = {
    "prompt": ["What is AI?", "Another prompt"],
    "chosen": ["AI stands for Artificial Intelligence...", "Another Chosen"],
    "rejected": ["AI is a vegetable.", "Another response"],
}

dataset = Dataset.from_dict(data)
print(dataset[1])

# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
#train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", logging_steps=10)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=dataset)
trainer.train()
"""

##############################################################################
