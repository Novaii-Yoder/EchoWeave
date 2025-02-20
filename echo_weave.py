from openai import OpenAI # type: ignore
from openaicom import convert_to_kg, api_key
from path_algos import a_star_search, path_to_nodes_and_relationships

from transformers import GPT2TokenizerFast # type: ignore
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.embeddings import OpenAIEmbeddings # type: ignore
from langchain.chains.question_answering import load_qa_chain # type: ignore
from langchain.schema import Document, SystemMessage, HumanMessage # type: ignore
from langchain_community.chat_models import ChatOpenAI # type: ignore
from requests.exceptions import ChunkedEncodingError # type: ignore


import numpy as np # type: ignore
import networkx as nx # type: ignore
import json
import re
import time
import os
import matplotlib.pyplot as plt # type: ignore

class EchoWeave:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.files = []
        self.entry_points = []
        self.embedding_model = OpenAIEmbeddings()
        self.keep_old_nodes = True

    def set_api_key(self, apikey):
        self.__api_key = apikey
        api_key(apikey)

    def __chunk_file(self, file):
        if os.path.splitext(file)[1] == ".pdf":
            return EchoWeave.__chunk_text(file)
        elif os.path.splitext(file)[1] == ".json":
            with open(file, 'r') as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of objects.")

            documents = []
            for entry in data:
                if 'image' in entry and 'description' in entry:
                    formatted_string = f"Image: {entry['image']}\nDescription: {entry['description']}"
                    documents.append(Document(page_content=formatted_string))
                else:
                    print(f"Skipping entry without 'image' or 'description': {entry}")
            return documents
        else:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()

            # Step 3: Create function to count tokens
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

            def count_tokens(text: str) -> int:
                return len(tokenizer.encode(text))

            # Step 4: Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size = 1024,
                chunk_overlap  = 32,
                length_function = count_tokens,
            )

            return text_splitter.create_documents([text])
          
    def __chunk_text(self, document):
        # Advanced method - Split by chunk

        # Step 1: Convert PDF to text
        import textract # type: ignore
        doc = textract.process(document)

        # Step 2: Save to .txt and reopen (helps prevent issues)
        with open("temp.txt", 'w') as f:
            f.write(doc.decode('utf-8'))

        with open("temp.txt", 'r') as f:
            text = f.read()

        # Step 3: Create function to count tokens
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text))

        # Step 4: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 1024,
            chunk_overlap  = 32,
            length_function = count_tokens,
        )

        return text_splitter.create_documents([text])

    def print_stats(self):
        print(f"""Knowledge Graph stats:
            \n   Number of nodes: {len(self.graph)}
            \n   Number of relations: {self.graph.size()}
            \n   Files: {self.files}
            \n   Entry points: {self.entry_points}
              """)
        
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(self.graph, k=0.5, iterations=100)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=300, font_size=5)
        plt.title("EchoWeave Visualization")
        plt.show()

    def remove_invisible_node(self, node):
        if not self.graph.nodes[node]: # empty node
            self.graph.remove_node(node)
        return

    def remove_reference(self, path):
        
        file_node = self.graph.nodes[path]
        #print(file_node)
        chunk_nodes = self.graph.neighbors(file_node["label"])
        related_topics = {}
        chunks_to_remove = []
        for chunk in chunk_nodes:
            related_topics = (self.graph.neighbors(chunk))
            for node in related_topics:
                self.graph.nodes[node]['references'][file_node['label']].remove(int(chunk.replace('Chunk ', '')))
                self.graph.nodes[node]['embedding'] = np.subtract(np.array(self.graph.nodes[node]['embedding']), np.array(self.graph.nodes[chunk]['embedding'])).tolist()

                # if a node has no references remove it
                if len(self.graph.nodes[node]['references'][file_node['label']]) == 0:
                    self.graph.nodes[node]['references'].pop(file_node['label'])
                if len(self.graph.nodes[node]["references"]) == 0:
                    chunks_to_remove.append(node)
                print(node)
                print(self.graph.nodes[node]["references"])
            chunks_to_remove.append(chunk)
        for chunk in chunks_to_remove:
            n = self.graph.neighbors(chunk)
            print(f"n = {n}")
            self.graph.remove_node(chunk)
            for ni in n:
                self.remove_invisible_node(ni)

        self.graph.remove_node(file_node["label"])
        self.files.remove(file_node["label"])
        return

    def update_file(self, path):
        if path not in self.files:
            raise Exception(f"File doesn't exist in EchoWeave, try add_file(\"{path}\")")
        
        # would a user ever want to keep the old file?
        # is there ever a situation where you want to remove all the nodes connected the file?
        #if not self.keep_old_nodes:
        #    self.remove_old_file(path)
        #    self.add_file(path)
        #else:
        self.remove_reference(path)
        self.add_file(path)
        return

    def add_file(self, path):
        if path in self.files:
            self.update_file(path)
            return
        if not os.path.isfile(path):
            raise Exception(f"Cannot find file {path}")
        
        self.files.append(path)
        paragraphs = EchoWeave.__chunk_file(self, path)
        i = 0
        size = len(paragraphs)
        kg = []

        summed_embedding = np.empty(1536)
        #full_text = ""
        for paragraph in paragraphs:
            i += 1
            print(f'Converting chunk {i}/{size} in {path}')
            k = EchoWeave.__convert_to_kg_plus(self, paragraph, i, path)
            if k is None:
                continue

            # add each chunk and connect them to the file
            embedding = self.__create_embedding_for_chunk(paragraph.page_content)
            #full_text += paragraph.page_content
            self.graph.add_node(f"Chunk {i}", label=f"Chunk {i}", type="chunk", embedding=embedding, text=paragraph.page_content)
            self.graph.add_edge(path, f"Chunk {i}", type="has_chunk") # maybe change to add_edges_from for speed

            
            #print(k)
            kg.append([k, embedding])
            summed_embedding = np.add(summed_embedding, embedding)
        
        # add the OG file to the graph
        self.graph.add_node(path, label=path, type="file", embedding=summed_embedding.tolist())

        #############################
        # This test showed that the sum of chunk embeddings has a very similar result to a full text embedding
        #full_embedding = self.embedding_model.embed_query(full_text)
        #print("Summed embedding = ", summed_embedding)
        #print("Full embedding  =  ", full_embedding)
        #print("cosine similarity = ", EchoWeave.cosine_similarity(summed_embedding, full_embedding))
        #############################

        for gra, chunk_embedding in kg:

            if gra is None:
                continue
            # Convert the JSON string to a Python dictionary if it's a string
            if isinstance(gra, str):
                try:
                    gra = json.loads(gra)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

            EchoWeave.__append_to_graph(self, gra, chunk_embedding)

        return

    def __convert_to_kg_plus(self, paragraph, paragraph_num, file_name, max_retries=5):
        retries = 0
        while retries < max_retries:
            try:
                k = convert_to_kg(paragraph)
                # Attempt to load the JSON
                k = EchoWeave.__clean_json_output(k)
                data = json.loads(k)
                for entity in data['entities']:
                    entity['file_name'] = file_name
                    entity['chunk_number'] = paragraph_num
                    #entity['chunk'] = paragraph
                return data

            except ChunkedEncodingError as e:
                retries += 1
                print(f"ChunkedEncodingError occurred. Retry {retries}/{max_retries}.")
                time.sleep(retry_delay) # type: ignore

            except json.JSONDecodeError as e:
                # If there's a JSON error, print a helpful error message and return None
                print(f"Error decoding JSON for paragraph {paragraph_num}: {e}")
                #print(k)
                return None

        print("Max retries reached. Could not complete the request, moving on.")
        return None

    def __append_to_graph(self, data, chunk_embedding):
        # Add nodes (entities) to the graph, along with metadata
        #print(data)
        for entity in data['entities']:
            node_id = entity['id']

            # Extract file_name and paragraph_number to only store them in the 'references' map
            file_name = entity['file_name']
            paragraph_number = entity['chunk_number']

            # Check if the node already exists in the graph
            if self.graph.has_node(node_id):
                # If the node exists, update its references map
                node_data = self.graph.nodes[node_id]
                references = node_data.get('references', {})
                

                # Update the reference map with the current file and paragraph number
                if file_name in references:
                    references[file_name].append(paragraph_number)
                else:
                    references[file_name] = [paragraph_number]

                # update embedding to include new chunk
                node_data['embedding'] = np.add(np.array(node_data['embedding']), np.array(chunk_embedding)).tolist()
                # Set the updated references map
                node_data['references'] = references
            else:
                # If the node doesn't exist, create it with metadata including references
                references = {file_name: [paragraph_number]}
                entity['references'] = references
                entity['embedding'] = chunk_embedding
                del entity['file_name']
                del entity['chunk_number']

                # Add the node with its metadata (without redundant file_name and paragraph_number)
                self.graph.add_node(node_id, **entity)
            self.graph.add_edge(f"Chunk {paragraph_number}", node_id, type="referenced")

        # Add edges (relationships) to the graph
        for relationship in data['relationships']:
            source = relationship['source']
            target = relationship['target']
            self.graph.add_edge(source, target, **relationship)  # Add relationship metadata as edge attributes

    def __create_embedding_for_chunk(self, chunk):
        embedding = self.embedding_model.embed_query(str(chunk))
        return embedding

    def __create_embedding_for_node(self, references, loaded_text_chunks):
        """Generate an embedding based on the references and the loaded text chunks."""
        # Collect the relevant text chunks for this node
        node_texts = []
        for file_name, reference in references.items():
            for ref in reference:
                chunk = loaded_text_chunks[file_name][int(ref) - 1]
                node_texts.append(chunk.page_content)

        # Concatenate all text chunks into one single string
        concatenated_text = " ".join(node_texts)

        if not concatenated_text:
            return None  # In case the node has no valid references

        # Use OpenAIEmbeddings to generate the embedding
        embedding = self.embedding_model.embed_query(concatenated_text)
        return np.array(embedding)  # Convert the embedding to a NumPy array for easy similarity calculations later

    def __clean_json_output(response_text):
        """
        Cleans the JSON output by removing unnecessary triple quotes or formatting.
        """
        # If the response starts and ends with triple quotes, remove them
        if response_text.startswith("'''") and response_text.endswith("'''"):
            response_text = response_text[3:-3].strip()

        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()

        return response_text

    def save_to_file(self, filename):
        data_to_save = {
            "files": self.files,
            "entry_points": self.entry_points,
            "graph": nx.node_link_data(self.graph), 
        }
        #print(data_to_save)
        # Save to JSON file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4)
    
    def search_documents_by_embedding(self, embedding, top_k=5):
        return self.__search(embedding, self.files, top_k)
    
    def search_chunks_by_embedding(self, embedding, top_k=5):
        chunks = []
        for file in self.files:
            print(self.graph.neighbors(file))
            for f in self.graph.neighbors(file):
                chunks.append(f)
        return self.__search(embedding, chunks, top_k)
    
    def __search(self, embedding, nodes_to_search, top_k=5):
        node_similarities = []
        for node in nodes_to_search:
            
            full_node = self.graph.nodes[node]
            node_embedding = np.array(full_node['embedding'])
            # Skip nodes that don't have embeddings
            if node_embedding is None:
                continue
        
            # Calculate cosine similarity between query embedding and node's embedding
            similarity = EchoWeave.cosine_similarity(embedding, node_embedding)
        
            # Add the node, and similarity score to the list
            node_similarities.append((full_node, similarity))

        # Sort the nodes by similarity score in descending order
        node_similarities = sorted(node_similarities, key=lambda x: x[1], reverse=True)
        return node_similarities[:top_k]


    def search_documents_by_query(self, query, top_k=5):
        embedding = self.embedding_model.embed_query(query)
        return self.search_documents_by_embedding(embedding=embedding, top_k=top_k)
    
    def search_chunks_by_query(self, query, top_k=5):
        embedding = self.embedding_model.embed_query(query)
        return self.search_chunks_by_embedding(embedding=embedding, top_k=top_k)
    
    # Function to calculate cosine similarity between two embeddings
    def cosine_similarity(embedding1, embedding2):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)

    @classmethod
    def load_from_file(cls, filename):
        """
        Loads the class instance from a saved JSON file.

        Args:
            filename (str): Path to the JSON file.

        Returns:
            EchoWeave: A new instance with loaded attributes.
        """
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create a new instance of the class
        instance = cls()
        instance.files = data.get("files", [])
        instance.entry_points = data.get("entry_points", [])

        # Convert JSON graph data back into a NetworkX graph
        instance.graph = nx.node_link_graph(data["graph"], edges="links") # edges are called links in old version

        print(f"Graph and attributes loaded from {filename}.")
        return instance

