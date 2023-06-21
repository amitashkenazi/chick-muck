import pandas as pd
import openai.embeddings_utils as embeddings_utils
import re
import pickle
import os
from csv import writer

def data_embedding(embeddings_utils, model='text-embedding-ada-002'):
    print("reading csv")
    df_faq = pd.read_csv('data/faq.csv')
    print("normalizing text")
    df_faq['answer'] = df_faq["answer"].apply(lambda x : normalize_text(x))
    print("getting embedding")
    df_faq['answer_embedding'] = df_faq["answer"].apply(lambda x : embeddings_utils.get_embedding(x, engine = model))
    df_faq['question_embedding'] = df_faq["question"].apply(lambda x : embeddings_utils.get_embedding(x, engine = model))
    print("saving pickle")
    # save df_faq to pickle
    with open('data/df_faq.pickle', 'wb') as handle:
        pickle.dump(df_faq, handle, protocol=pickle.HIGHEST_PROTOCOL)

def normalize_text(s, sep_token = " \n "):
    print(s)
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    return s

# search through the reviews for a specific product
def search_faq(embeddings_utils, user_query, top_n=3, th=0.8, model='text-embedding-ada-002'):
    # get embedding for user query
    embedding = embeddings_utils.get_embedding(
        user_query,
        engine=model
    )
    print(f"embedding={len(embedding)}")
    save_question_embedding(user_query, embedding)
    #load pickle
    with open('data/df_faq.pickle', 'rb') as handle:
        df_faq = pickle.load(handle)
        # calculate cosine similarity between user query and all reviews
        df_faq["similarities"] = df_faq.answer_embedding.apply(lambda x: embeddings_utils.cosine_similarity(x, embedding))
        # remove all similarities below 0.8
        df_faq = df_faq[df_faq.similarities > th]

        # sort by similarity and return top n
        res = (
            df_faq.sort_values("similarities", ascending=False)
            .head(top_n)
        )
        print(f"knowledge_management.py search_faq res={res}")
        return res
    return None

# add question and answer to the faq csv
def add_question_answer(question, answer):
    # load csv
    df_faq = pd.read_csv('data/faq.csv')
    # append new question and answer
    df_faq = df_faq.append({"question": question, "answer": answer}, ignore_index=True)
    # save csv
    df_faq.to_csv('data/faq.csv', index=False)

# read embeddings vector from csv and return a list of embeddings
def read_embeddings():
    # load csv
    df_faq = pd.read_csv('data/faq.csv')
    # return list of embeddings
    return df_faq.question_embedding.to_list()


def save_question_embedding(question, embedding):
    csv_file = 'data/questions_embeddings.csv'
    
    with open(csv_file, 'a') as f_object:
 
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
    
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow([question, embedding])
    
        # Close the file object
        f_object.close()

# run if called directly
if __name__ == "__main__":
    # Set up OpenAI API key
    import openai
    openai.api_key = os.environ['OPENAI_API_KEY']
    openai.api_base = os.environ['OPENAI_ENDPOINT']
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"
    data_embedding(embeddings_utils, model='text-embedding-ada-002')
    