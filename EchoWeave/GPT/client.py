from .llama import Llama_Client
from .openai import Openai_Client
from .prompts import (
    kg_builder_system_message
)

class GPT_Client:
    allowed_clients = [Llama_Client, Openai_Client]

    def __init__(self):
        self.client = None

    def set_client(self, client):
        if type(client) not in self.allowed_clients:
            print("Client is not currently supported!")
            return -1
        if self.client != None:
            print("Changing the client after already starting building is not recommended.")
        self.client = client

    def set_api_key(self, apikey):
        try: 
            self.client.set_api_key(apikey)
        except:
            print("Failed to set api_key, client may not need it")

    def gpt_call(self, system_message, user_message):
        return self.client.gpt_call(system_message, user_message)
    
    def convert_to_kg(self, paragraph:str):
        system_message = kg_builder_system_message
        user_message = "Convert this paragraph to a Knowledge Graph json: " + str(paragraph)
        return self.client.gpt_call(system_message, user_message)

    def rag_query(self, query, docs, subgraph):
        system_message = "You are a helpful assistant that uses the context given to answer the question. If you are unsure say there is not enough information provided."
        user_message = query + " Given the document chunks: " + docs + " And the the relationship graph: " + subgraph
        return self.client.gpt_call(system_message, user_message)

