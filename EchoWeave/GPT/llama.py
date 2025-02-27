import ollama # type: ignore

#response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": "What is Llama?"}])
#print(response["message"])

class Llama_Client:
    def __init__(self, model="llama3.2"):
        self.model = model

    def gpt_call(self, system_cotent, user_content):
        if type(system_cotent) != str:
            system_cotent = str(system_cotent)
        if type(user_content) != str:
            user_content = str(user_content)

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "system", "content": system_cotent},
                       {"role": "user", "content": user_content}],
            stream=False
        )
        #print(response['message'])
        return response['message']['content']

    def convert_to_kg(self, textblock):
        
        if type(textblock) != str:
            textblock = str(textblock)


        response = ollama.chat(
            model=self.model,
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

                Convert this text to a knowledge graph JSON: """},
                {"role":"user", "content":"""Convert this text to a knowledge graph json: """ + textblock}
                ], stream=False
        )
        #print(response['message'])
        return response['message']['content']




