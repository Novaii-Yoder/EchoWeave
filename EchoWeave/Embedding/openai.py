from langchain_community.embeddings import OpenAIEmbeddings # type: ignore

class Openai_Embeddings:
    def __init__(self, model="text-embedding-ada-002", apikey = None):
        self.model = model
        if apikey == None:
            print("Don't forget the API key...")
        else:
            self.set_api_key(apikey)
        self.embedding = OpenAIEmbeddings(model=self.model)

    def set_api_key(self, key):
        self.api_key = key
        self.embedding.openai_api_key=key
        

    def embed_query(self, query):
        if type(query) != str:
            query = str(query)

        response = self.embedding.embed_query(query)

        return response