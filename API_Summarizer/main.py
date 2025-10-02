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

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client, create_client
from typing import List
from openai import OpenAI

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=openai_api_key)


# gpt-5-mini
# gpt-4.1-mini
llm = os.getenv('LLM_MODEL', 'gpt-4.1')
model = OpenAIModel("gpt-4.1", api_key=openai_api_key)

system_prompt_summary = """
You are an expert at Environmental earth science and you have access to all the documentation to,
including examples, an API reference, and other resources.


Always let the user know when you didn't find the answer in the documentation or the right URL - DO NOT MAKE YOUR OWN URLs. Only Use URLS from the 'url' key.
Always generate your result with an appropriate URL that you get from the following context, URLS, Summary and Title.

Ask follow-up questions if you do not have enough context. If the context is not clear ask try to answer with your own knowledge and also ask follow up questions.

"""

app = Flask(__name__)
api = Api(app)

client = OpenAI(api_key=openai_api_key)




def summarize_json(data, max_chars: int = 12000) -> str:

    pretty = json.dumps(data, ensure_ascii=False, indent=2)[:12000]
    msgs = [
        {"role":"assistant","content":"Summarize JSON for non-technical readers in 5-10 lines."},
        {"role":"user","content":f"```json\n{pretty}\n```"}
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.2, max_tokens=700)
    return resp.choices[0].message.content.strip()


# another resource to calculate the square of a number
class Summary(Resource):

    def get(self, text):

        summary = summarize_json(text)
        return jsonify({'summary': summary})


def chat_response(data, max_chars: int = 12000) -> str:

    #pretty = json.dumps(data, ensure_ascii=False, indent=2)[:12000]
    msgs = [
        {"role":"assistant","content":"Provide an appropriate and concise and in depth response for the following query"},
        {"role":"user","content":f"```{data}\n```"}
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.2, max_tokens=200)
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