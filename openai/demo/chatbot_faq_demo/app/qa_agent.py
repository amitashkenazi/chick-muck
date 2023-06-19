from pprint import pprint
def generate_user_input(messages_input, openai, model, temperature,max_tokens):
    # remove the value the "role": "system" from messages_input:
    pprint(f"generate_user_input messages_input={messages_input}")
    pprint(f"generate_user_input messages_input={type(messages_input)}")
    if len(messages_input) == 0:
        return "hello"
    messages = [
        {"role":"system","content":
         """You are a qa agent that test an Azure customer support agent whose primary goal is to help users with issues they are experiencing with their Azure Free accounts. 
         The assistant role consist of the responses from the support agent we are testing and the user role consist of previous massages from the qa agent (you are the qa agent that ask question the supprt agent).
         be inpatient and ask the agent to help you with your issue.
         """}
        ]
    for message in messages_input:
        if message["role"] == "assistant":
            messages.append({"role":"user","content":message["content"]})
        else:
            messages.append({"role":"assistant","content":message["content"]})
    messages.append({"role":"system","content":"generate a message from the user that continue the conversation"})
    print (f"generate_user_input messages={messages}")
    response = openai.ChatCompletion.create(
        engine=model,
        messages = messages,
        temperature=1.0,
        max_tokens=300,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    print(f"generate_user_input response={response.choices[0].message.content}")
    return response.choices[0].message.content
