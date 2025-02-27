import networkx as nx
import matplotlib.pyplot as plt 
import heapq
import networkx as nx
import matplotlib.pyplot as plt

from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
from numpy import dot
from numpy.linalg import norm
import openai # type: ignore
import sys

def visualize_graph(graph, figsize=(12, 10), title="EchoWeave Visualization", save_to_file=True, file_name="EchoWeave_Vis.jpg"):
    
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(graph, k=0.5, iterations=100)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=300, font_size=5)
    plt.title(title)
    plt.show()


# Similarity score for embeddings
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# Heuristic function for A*, cost between any 2 points
def heuristic(start, target):
    
    try:
        similarity = cosine_similarity(start["embedding"], target["embedding"])
    except:
        # Default value when error occurs
        return 0.5
    
    return 1 - similarity

def a_star_search(G, start, goal):
    s = G.nodes[start]
    g = G.nodes[goal]

    if s == None or g == None:
        raise Exception(f"Start or end node don't exist in graph. start: {start}, end: {goal}")

    # Priority queue for the open set
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # Dictionaries for cost and path
    from_start_cost = {start: 0}
    total_cost = {start: heuristic(s, g)}
    came_from = {start: None}
    
    while open_set:
        current_cost, current_node = heapq.heappop(open_set)
        
        if current_node == goal:
            return reconstruct_path(came_from, start, goal)
        
        for neighbor in G.neighbors(current_node):
            tentative_from_start_cost = from_start_cost[current_node] + G[current_node][neighbor].get('weight', 1)
            
            if neighbor not in from_start_cost or tentative_from_start_cost < from_start_cost[neighbor]:
                from_start_cost[neighbor] = tentative_from_start_cost
                n = G.nodes[neighbor]
                total_cost[neighbor] = tentative_from_start_cost + heuristic(n, g)
                heapq.heappush(open_set, (total_cost[neighbor], neighbor))
                came_from[neighbor] = current_node
    
    return None

def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def path_to_nodes_and_relationships(G, path):
    nodes_and_relationships = []
    for i in range(len(path) - 1):
        source = path[i]
        target = path[i + 1]
        relationship = G.get_edge_data(source, target)
        #print(relationship)
        nodes_and_relationships.append((source, relationship["type"], target))
    return nodes_and_relationships

# old function used to visually see the path in the graph, doesn't work well with large graphs
def visualize_graph_with_path(G, path):
    start = path[0]
    goal = path[-1]
    pos = nx.spring_layout(G)  # Position nodes using the spring layout
    
    plt.figure(figsize=(12, 12))

    # Draw all nodes and edges in grey
    nx.draw(G, pos, with_labels=True, node_color='grey', edge_color='grey', node_size=500, alpha=0.6, font_size=10)
    
    # Highlight nodes and edges in the path
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightblue', node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=2)
    nx.draw_networkx_labels(G, pos, labels={node: node for node in path}, font_color='black', font_size=10)
    
    # Highlight start and goal nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='lime', node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist=[goal], node_color='red', node_size=700)
    
    plt.title("Knowledge Graph with Highlighted Path")
    plt.show()


def api_key(key):
    openai.api_key = key


def convert_to_topic(question):
    """
    Extracts the main topic from a given question using OpenAI's GPT model.

    Args:
        question (str): The user question.

    Returns:
        str: A concise topic extracted from the question.
    """
    if not openai.api_key:
        raise ValueError("API key not set. Use openai.api_key = 'your_api_key'.")

    if not isinstance(question, str):
        question = str(question)

    response = openai.Client().chat.completions.create(
        model="gpt-4o",  # Use GPT-4o (or switch to another available model)
        messages=[
            {"role": "system", "content": """You are a helpful but concise assistant that extracts topics from questions. 
                Only return the main subject in a few words. Examples:
                user: "What can cause a car accident?"
                system: "Car accident"
                user: "What is a basilica used for?"
                system: "Basilica"
                user: "What temperature limit of a car engine is safe?"
                system: "Car engine"
                Convert this question: """},
            {"role": "user", "content": question}
        ]
    )

    # Extract and return the full response as a single string
    return response.choices[0].message.content.strip()

def ask_chat_gpt(question):
    if not openai.api_key:
        raise ValueError("API key not set. Use set_api_key() to set it.")

    if not isinstance(question, str):
        question = str(question)


    response = openai.Client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": """You are a helpful but concise assistant that answers questions simply and provides no explanation. For example:
            user: \"What can cause a car accident?\"
            system: Poor maintenance
            or user: \"What is a basilica used for?\"
            system: Religion
            Answer this question: """ + question}],
    )
    
    # Extract and return the full response as a single string
    return response.choices[0].message.content.strip()





def convert_path_to_sentence(path):
    if not openai.api_key:
        raise ValueError("API key not set. Use set_api_key() to set it.")

    if type(path) != str:
        path = str(path)

    response = openai.Client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": """You are a helpful but concise assistant that converts an array of objects connected by relationships. You will use the array given to you to draw a logical path of reasoning from the start to end. You will return an explanation that is concise and makes logical sense connecting each element in the array given to you. For example:
            user: \"('Seats', 'has', 'Car'), ('Car', 'has', 'Engine'), ('Engine', 'needs', 'Fuel'), ('Fuel', 'type', 'Gas')\"
            system: All car's have an engine, engines require fuel to run, and gas is a type of fuel.
            Convert this path: """ + path}],
    )
    
    # Extract and return the full response as a single string
    return response.choices[0].message.content.strip()

