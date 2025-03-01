from .llama import Llama_Embeddings
from .openai import Openai_Embeddings


class Embedding_Client:
    allowed_clients = [Llama_Embeddings, Openai_Embeddings]

    def __init__(self):
        self.client = None

    def set_embedding(self, client):
        if type(client) not in self.allowed_clients:
            print("Embedding client is not currently supported!")
            return -1
        if self.client != None:
            print("Changing the client after already starting building is not recommended. A complete graph embedding change needs to happen.")
        self.client = client

    def set_api_key(self, apikey):
        try: 
            self.client.set_api_key(apikey)
        except:
            print("Failed to set api_key, client may not need it")

    def embed_query(self, query):
        return self.client.embed_query(query)