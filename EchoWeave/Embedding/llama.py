import ollama # type: ignore

class Llama_Embeddings:
    def __init__(self, model="mxbai-embed-large"):
        self.model = model

    def embed_query(self, query):
        if type(query) != str:
            query = str(query)

        response = ollama.embed(
            model=self.model,
            input=query,
        )
        #print(type(response["embeddings"][0]))
        return response["embeddings"][0]