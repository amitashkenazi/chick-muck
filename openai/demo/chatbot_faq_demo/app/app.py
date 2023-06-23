from streamlit_chat import message
import streamlit as st
import openai
import openai.embeddings_utils as embeddings_utils
from pprint import pprint
import qa_agent
import knowledge_management
import analytics
import os

# Set up OpenAI API key
openai.api_type = "azure"
openai.api_base = os.environ['OPENAI_ENDPOINT']
openai.api_version = "2023-03-15-preview"
openai.api_key = os.environ['OPENAI_API_KEY']

global_user_context = {"content": []}

messages_history = []


def chat(model, model_embedding):
    """
    This function facilitates an interactive chat session between a user and a 
    chatbot using Streamlit's interface. 
    The chatbot utilizes a given language model and corresponding embeddings. 
    The function provides controls to set the maximum number of tokens in responses, 
    randomness of responses, and to enable or disable embeddings mode. 
    The conversation's history is maintained in session variables, 
    allowing the model to generate contextually relevant responses.
    """
    print("chat")
    max_tokens = st.slider('max tokens',100,4000)
    temperature = st.slider('temperature',0.0,1.0)
    
    agent_mode = st.checkbox('Embeddings mode')
    prompt_conf = st.checkbox('configured - start conversation')
    
    if prompt_conf:
        # session state initialization
        st.success('prompt configuration saved')        
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        if 'embedding_answers' not in st.session_state:
            st.session_state['embedding_answers'] = []
        if 'past' not in st.session_state:
            st.session_state['past'] = []
        if 'message_history' not in st.session_state:
            st.session_state['message_history'] = []
        
        user_input=st.text_input("You:",key='input')
        if st.button('Generate User message', key='Continue'):
            user_input = qa_agent.generate_user_input(messages_input=st.session_state['message_history'], openai=openai, model=model, temperature=temperature, max_tokens=max_tokens)

        if user_input:
            print("user input: ",user_input)
            output=generate_gpt_chat(prompt=user_input,model=model,model_embedding=model_embedding, max_tokens=max_tokens,temperature=temperature, agent_mode=agent_mode)
            global_user_context["content"].append(user_input)
            #store the output
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output["res"])
            if agent_mode:
                embedding_answers = "The answers from the FAQ database are:\n"
                for idx, answer in enumerate(output["answers"]["answer"]):
                    embedding_answers += f"""{idx}. {answer}\n"""
                st.session_state['embedding_answers'].append(embedding_answers)
            st.session_state['message_history'].append({"role":"user","content":user_input})
            st.session_state['message_history'].append({"role":"assistant","content":output["res"]})

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                if agent_mode:
                    message(st.session_state["embedding_answers"][i], key=str(i) + '_embedding')
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  



def generate_gpt_chat(prompt,model='gpt-35-turbo', model_embedding='text-embedding-ada-002', max_tokens=4000,temperature=0.5, agent_mode=False):
    """
    This function generates a chat response using the OpenAI's GPT model. 
    It operates in two modes: open chat and FAQ mode. In open chat, 
    it takes user's input and generates a response based on pre-defined role-play scenarios. 
    In FAQ mode, it searches a knowledge base for answers to the user's query and generates a 
    response combining the relevant answers. 
    The function returns the generated response and, in FAQ mode, the relevant answers from the knowledge base.
    """
    answers = []
    if not agent_mode:
        print("open chat mode")
        system_message = get_system_message()
        few_shots_string = ""
        for few_shot in get_few_shots():
            few_shots_string += f"""{few_shot}"""
        system_message += few_shots_string
        print(f"system message: {system_message}")
        messages = [ { "role":"system","content":system_message}]
        for msg in st.session_state['message_history']:
            messages.append(msg)
        messages.append({"role":"user","content":prompt})
        # messages.append({"role":"system","content":"Answer only questions that are related to Azure free account or AKS service. if the user asked a question that is not related to Azure free account or AKS service, apologize and ask the user to ask a different question or contact support"})  
    else:
        print("FAQ mode")
        answers = knowledge_management.search_faq(embeddings_utils, prompt, model=model_embedding)
        answers_string = ""
        for idx, answer in enumerate(answers['answer']):
            answers_string += f"""{idx}. {answer}"""
        if len(answers) == 0:
            prompt = f"""The user input: '{prompt}'. If the user did not ask anything, respond and encourage him to share his problem. if the user asked a question, We could not find any answers in the database. please appologize and ask the user to ask a different question or contact support"""
        else:
            prompt = f"""The user input: {prompt}. If the user did not ask anything, respond and encourage him to share his problem. if the user asked a question, The relevant answers are: 
            {answers_string}. combine the answers to a single answer and return it to the user. use only the facts in the answers and do not add any new information"""
        system_message = get_system_message()
        few_shots_string = ""
        for few_shot in get_few_shots():
            few_shots_string += f"""{few_shot}"""
        system_message += few_shots_string
        
        messages = [ { "role":"system","content":system_message}]
        for msg in st.session_state['message_history']:
            messages.append(msg)
        messages.append({"role":"user","content":prompt})
        # messages.append({"role":"system","content":"Answer only questions that are related to Azure free account or AKS service. if the user asked a question that is not related to Azure free account or AKS service, apologize and ask the user to ask a different question or contact support"})
    print(f"messages for the chatbot: {messages}")
    print(f"model: {model}")
    response = openai.ChatCompletion.create(
        engine=model,
        messages = messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return {
        "res": response.choices[0].message.content,
        "answers": answers
        }

def get_system_message():
    return """You are an agent that returns answer based on the knowledge base.
you will get the answers relevant to the questions, and you will return the answer that is most relevant to the question.
You are an Azure customer support agent whose primary goal is to help users with issues they are experiencing while using azure.
You are friendly and concise. You only provide factual answers to queries, and provide answers that are related to Azure.
In the event of a query about Azure services, your responses should be strictly confined to the AKS service, also known as Azure Kubernetes Service or AKS. 
If a question pertains to any other Azure service beyond AKS, courteously apologize and prompt the user to either pose a different question or reach out to customer support.
you will not return answers that are not related to the question.
answer only questions related to Azure.
Few-shots examples:
The following are examples of how you should respond to the user:
"""

def get_few_shots():
    return [
        [
            {"role":"user", "content":"How much credit do you get when Subscribing to AWS?"},
            {"role":"assistant", "content":"I apologize, but I do not have information about the prices of other cloud platforms such as AWS. My primary focus is to assist with issues regarding Azure cloud platform. Is there a specific issue you are having with your Azure account that I may be able to help with?"}
        ],
        [
            {"role":"user", "content":"How much credit do you get when Subscribing to GCP?"},
            {"role":"assistant", "content":"I apologize, but I do not have information about the prices of other cloud platforms such as GCP. My primary focus is to assist with issues regarding Azure cloud platform. Is there a specific issue you are having with your Azure account that I may be able to help with?"}
        ],
        [
            {"role":"user", "content":"Does key vault is multi-region service"},
            {"role":"assistant", "content":"I apologize, but I do not have information about other services other than AKS. My primary focus is to assist with issues regarding Azure Free Acount or AKS service. Is there a specific issue you are having with your Azure account that I may be able to help with?"}
        ] 
         
    ]

model = st.text_input("select chat model",key='model', value='gpt-35-turbo')
model_embedding = st.text_input("select chat model",key='model_embedding', value='text-embedding-ada-002')
    
if st.button('Create Embeddings from FAQs', key='embeddings'): 
    knowledge_management.data_embedding(embeddings_utils, model=model_embedding)



chat(model, model_embedding)

if st.button('Questions Clustering', key='cluster'):
    # Display the plot using Streamlit
    fig, df, colors, num_clusters = analytics.analysis()
    st.pyplot(fig)

    # Print questions beneath the plot, colored by cluster
    st.write("\nClustered Questions:\n")
    for i in range(num_clusters):
        st.write(f"Cluster {i+1} (color = {colors[i]}):")
        for question in df['question'][df['cluster'] == i].values:
            st.write(question)
    