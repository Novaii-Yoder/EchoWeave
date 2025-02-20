import openai # type: ignore
import sys

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



##print(convert_path_to_sentence(sys.argv[1]))

def convert_to_kg(textblock):
    if not openai.api_key:
        raise ValueError("API key not set. Use set_api_key() to set it.")

    if type(textblock) != str:
        textblock = str(textblock)


    response = openai.Client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": """You are a helpful assistant that converts a block of text into a JSON file knowledge graph that contains the main topics and relationships from the text. Use 'entities' and 'relationships' as the JSON sections. 
            Do NOT include triple quotes (''') or code blocks.
            Output ONLY the JSON without additional formatting 
            
            for example:
            user: \"In the bustling city of Venora, nestled between the technological hub of Mardale and the historic district of Eastgate, Ella discovered a mysterious device in an old electronics shop, marked as an antique from the era of steam engines but equipped with advanced quantum circuitry. As she delved into its secrets, she found it could decrypt encrypted communications from the neighboring city's underground movement, a group aiming to disrupt the monopoly of TechGiant Corp over the city’s tech market. Her alliance with the movement not only threatened her safety but also poised her as a key figure in a burgeoning tech revolution, drawing the attention of both the rebels and corporate spies.\"
            system: 
{
  "entities": [
    {
      "id": "Ella",
      "type": "Character",
    },
    {
      "id": "Venora",
      "type": "City",
    },
    {
      "id": "Mardale",
      "type": "City",
    },
    {
      "id": "Eastgate",
      "type": "District",
    },
    {
      "id": "Mysterious Device",
      "type": "Technology",
    },
    {
      "id": "Underground Movement",
      "type": "Group",
    },
    {
      "id": "TechGiant Corp",
      "type": "Corporation",
    },
    {
      "id": "Rebels",
      "type": "Group",
    },
    {
      "id": "Corporate Spies",
      "type": "Group",
    }
  ],
  "relationships": [
    {
      "source": "Ella",
      "target": "Mysterious Device",
      "type": "found"
    },
    {
      "source": "Mysterious Device",
      "target": "Venora",
      "type": "located_in"
    },
    {
      "source": "Venora",
      "target": "Mardale",
      "type": "neighbor"
    },
    {
      "source": "Venora",
      "target": "Eastgate",
      "type": "includes"
    },
    {
      "source": "Mysterious Device",
      "target": "Underground Movement",
      "type": "decrypts_communications_for"
    },
    {
      "source": "Underground Movement",
      "target": "TechGiant Corp",
      "type": "opposes"
    },
    {
      "source": "Ella",
      "target": "Underground Movement",
      "type": "allies_with"
    },
    {
      "source": "Ella",
      "target": "Rebels",
      "type": "attracts_attention_from"
    },
    {
      "source": "Ella",
      "target": "Corporate Spies",
      "type": "attracts_attention_from"
    }
  ]
}

            Convert this text to a knowledge graph JSON: """ + textblock}],
    )
    # Extract and return the full response as a single string
    #print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()


#print(convert_to_kg("Lemons are good"))
#print(convert_to_kg("In the bustling city of Venora, nestled between the technological hub of Mardale and the historic district of Eastgate, Ella discovered a mysterious device in an old electronics shop, marked as an antique from the era of steam engines but equipped with advanced quantum circuitry. As she delved into its secrets, she found it could decrypt encrypted communications from the neighboring city's underground movement, a group aiming to disrupt the monopoly of TechGiant Corp over the city’s tech market. Her alliance with the movement not only threatened her safety but also poised her as a key figure in a burgeoning tech revolution, drawing the attention of both the rebels and corporate spies."))
