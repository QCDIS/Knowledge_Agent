from __future__ import annotations as _annotations

# using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api

import os, json, requests


from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

#from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
#from supabase import Client, create_client
from typing import List
from openai import OpenAI

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=openai_api_key)


# gpt-5-mini
# gpt-4.1-mini
llm = os.getenv('LLM_MODEL', 'gpt-4.1')
#model = OpenAIModel("gpt-4.1", api_key=openai_api_key)
model = OpenAIModel("gpt-4.1")
system_prompt_summary = """
You are an expert at Environmental earth science and you have access to all the documentation to,
including examples, an API reference, and other resources.

Summarize the keyword in details in 7-8 lines in terms of environment.


Your answer will be printed as a text. So do not print any extra or unnecessary characters in your summary.
"""

system_prompt_chat = """
You are an expert at Environmental earth science and you have access to all the documentation to,
including examples, an API reference, and other resources.

Based on the user query provide an appropriate response to the question.

Also suggest one (only one) of these links suggesting further response.
Chat based system: https://chat-envri.qcdis.org/
Classical Search System: https://search-envri.qcdis.org/search/
Catalogue of Search: https://catalogue.staging.envri.eu/

Your answer will be printed as a text. So do not print any extra or unnecessary characters in your summary.
"""

# Your input is a Json file. Use the "pageContetnts" keys in the JSON file to summarize the response.
# Summarize JSON for non-technical readers.

app = Flask(__name__)
api = Api(app)

client = OpenAI(api_key=openai_api_key)




def summarize_json(data, max_chars: int = 100000) -> str:

    print(data)
    pretty = json.dumps(data, ensure_ascii=False, indent=2)[:max_chars]
    #print(pretty)
    msgs = [
        {"role":"assistant","content":system_prompt_summary},
        {"role":"user","content":f"```json\n{pretty}\n```"}
    ]
    resp = client.chat.completions.create(model="gpt-4.1", messages=msgs, temperature=0.2, max_tokens=2000)
    return resp.choices[0].message.content.strip()


# another resource to calculate the square of a number
class Summary(Resource):

    def get(self, text):

        summary = summarize_json(text)
        return jsonify({'summary': summary})


def chat_response(data, max_chars: int = 12000) -> str:

    #pretty = json.dumps(data, ensure_ascii=False, indent=2)[:12000]
    msgs = [
        {"role":"assistant","content":system_prompt_chat},
        {"role":"user","content":f"```{data}\n```"}
    ]
    resp = client.chat.completions.create(model="gpt-4.1", messages=msgs, temperature=0.2, max_tokens=200)
    return resp.choices[0].message.content.strip()


class Chat(Resource):
    def get(self, text):

        reply = chat_response(text)
        return jsonify({'summary': reply})



# adding the defined resources along with their corresponding urls
api.add_resource(Summary, '/summary/<string:text>')
api.add_resource(Chat, '/chat/<string:text>')

# driver function
if __name__ == '__main__':

    #app.run(debug = True)
    app.run(host="0.0.0.0", port=5000, debug=False)
