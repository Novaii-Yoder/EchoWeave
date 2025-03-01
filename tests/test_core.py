import unittest
import os
from EchoWeave.core import EchoWeave
import networkx as nx
from EchoWeave.GPT.llama import Llama_Client
from EchoWeave.Embedding.llama import Llama_Embeddings

class TestEchoWeave(unittest.TestCase):
    
    def setUp(self):
        apikey_file = "../../serviceaccount.apikey"

        file = open(apikey_file, 'r')
        apikey = file.read().strip()
        file.close()
        os.environ["OPENAI_API_KEY"] = apikey
        
        self.obj = EchoWeave()

    def test_instance_creation(self):
        self.assertIsInstance(self.obj, EchoWeave)
    
    def test_graph_creation(self):
        
        apikey_file = "../../serviceaccount.apikey"

        file = open(apikey_file, 'r')
        apikey = file.read().strip()
        file.close()
        os.environ["OPENAI_API_KEY"] = apikey
        
        self.obj.set_api_key(apikey)

        self.obj.add_file("test.txt")

        self.assertIsInstance(self.obj.graph, nx.Graph)
        self.assertIsInstance(self.obj.files, list)
        self.assertEqual(len(self.obj.files), 1)
    
    
    
    def test_llama(self):


        apikey_file = "../../serviceaccount.apikey"
        file = open(apikey_file, 'r')
        apikey = file.read().strip()
        file.close()
        os.environ["OPENAI_API_KEY"] = apikey

        self.obj.set_api_key(apikey)

        self.obj.set_client(Llama_Client(model="llama3.2"))
        self.obj.set_embedding(Llama_Embeddings(model="mxbai-embed-large"))
        #self.obj.set_embedding()
        #response = self.obj.client.gpt_call(
        #   "You are a helpful assistant that answers quesitons simply.",
        #   "What is the meaning of life?")
        #print(response)
        
        self.obj.add_file("notes.docx")
        self.obj.save_to_file("notes.json")
        self.obj.print_stats()
        """
    def test_llama_search(self):
        self.obj = self.obj.load_from_file("notes.json")
        self.obj.set_client(Llama_Client(model="llama3.2"))
        self.obj.set_embedding(Llama_Embeddings(model="mxbai-embed-large"))
        #self.obj.print_stats()

        ret = self.obj.RAG("How did the encounter with Drad go for the party members?")
        print(ret)
        """
        


if __name__ == "__main__":
    #unittest.main()
    unittest.main()