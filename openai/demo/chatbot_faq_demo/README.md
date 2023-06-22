This project provides an Azure support chatbot which leverages OpenAI's GPT-3.5-turbo model to interact with users, answer their queries, and provide support related to Azure services.

Overview
The chatbot has two modes:

Open Chat Mode: In this mode, the chatbot emulates a customer support agent and provides factual answers to queries related to Azure.

FAQ Mode: In this mode, the chatbot leverages a knowledge base to provide responses. It first searches the knowledge base for relevant answers to the user's question, combines the answers into a single response, and returns it to the user. In this mode, the chatbot only answers questions related to Azure Kubernetes Service (AKS) and advises the user to ask a different question or contact support for questions related to other services.

Dependencies
This project relies on the following dependencies:

streamlit
openai
streamlit-chat
matplotlib
scikit-learn
plotly

You can install the dependencies using pip:
pip install -r requirements.txt

You need to have these dependencies installed on your system or virtual environment before running the chatbot.

How to Run the Chatbot
Clone the Git repository.
Set up your OpenAI API key and endpoint in your environment variables.
Run streamlit run your_filename.py in your terminal.
After running the command, a Streamlit application will open in your web browser where you can interact with the chatbot.

Configuration
In the Streamlit application, you can configure the following parameters:

Chat model: The OpenAI model used for chat. The default is 'gpt-3.5-turbo'.
Chat model for embedding: The OpenAI model used for embedding. The default is 'text-embedding-ada-002'.
Max tokens: The maximum number of tokens for the chat model to produce. It can be between 100 and 4000.
Temperature: The randomness of the chat model's output. A value closer to 1 makes the output more random, while a value closer to 0 makes it more deterministic.
Embeddings mode: This checkbox determines whether the chatbot operates in FAQ mode
Configured - start conversation: This checkbox must be checked to start the chatbot.
User Bot: This button generates a user message based on the conversation history. It's useful for simulating a conversation.
Note

This chatbot is a prototype and is not intended for commercial use. It's designed to showcase the capabilities of OpenAI's language model in the context of customer support.

***using the dockerfile***

To build the docker:
docker build -t [tag name] .

to run the docker:
docker run -e OPENAI_API_KEY=[your api key] -e OPENAI_ENDPOINT=[your host endpoint]  -p 8501:80 -it [tag name]