~~~
  _____     _        __        __                   
 | ____|___| |__   __\ \      / /__  __ ___   _____ 
 |  _| / __| '_ \ / _ \ \ /\ / / _ \/ _` \ \ / / _ \
 | |__| (__| | | | (_) \ V  V /  __/ (_| |\ V /  __/
 |_____\___|_| |_|\___/ \_/\_/ \___|\__,_| \_/ \___|
~~~
 A all-in-one Knowledge Graph option for storing and retrieving textual data while keeping symantic connections.

 EchoWeave is a continuation of a research project that can be found on my GitHub conducted as part of my Master's degree at NMSU. I wanted to flesh out the idea and build a knowledge graph tool that could be used on hopefully larger graphs and constantly updating sources.

 The main personal use for EchoWeave is to host a "second brain" a place where I can store large amounts of information I've come across or created and have an easy way to search and find relavent information that I have already seen and understood. I.E. I wanted this to store every textbook I read, every note I've taken, every passing thought, and then be able to draw connections between them so I can later search the "second brain" to find related topics and notes to whatever I am currently thinking about or working on. 

 The project has also shifted use to help Game Master's in RPGs like Dungeons and Dragons, as you can store all your notes, descriptions, events, and history of the game and easily find answers to specific questions like "Which elf was the cause of the fall of Alish kingdom?" This use case even has the ability to use the graph for Retrieval Augmented Generation (RAG) for LLMs, which allow users to essentially have secretary in the DMing that has a really deep understanding of events and connections between people, events, and places in the game. 

 # About EchoWeave internals (for the nerds)
 EchoWeave is essentially a wrapper for a knowledge graph library in python called NetworkX, however it does include many features that hopefully increase the usefulness for retrieval purposes. It uses embeddings from LLM's to capture the symantic meaning of the documents and builds the knowledge graph with these embeddings in mind so that they can be searched by the embedding and documents, concepts, and relationships related to the query can be found.

# How to use EchoWeave
EchoWeave assumes you understand a decent amount of python as it is a class that gives you tools to build and maintain the knowledge graph, but doesn't do it completely automatically (yet). 

(WIP) A tutorial or demo script 
