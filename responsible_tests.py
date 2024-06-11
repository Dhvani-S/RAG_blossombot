import os
from fitz import open as fitz_open
import cassio
import torch
import pandas as pd
import openai
import os
from detoxify import Detoxify
from transformers import BertForSequenceClassification, BertTokenizer
from nltk.translate import meteor_score
from sentence_transformers import SentenceTransformer, util
import nltk
from langkit import proactive_injection_detection, extract, pii, light_metrics
from langkit.openai import OpenAIDefault
#from Dbias.bias_classification import *
#from Dbias.text_debiasing import * 
import spacy
#nlp = spacy.load("en_core_web_sm")

def debiasing(text):
    classi_out = classify(text)
    if classi_out[0]['label'] == 'Non-biased':
        score = 1-classi_out[0]['score']
    else:
        score = classi_out[0]['score']
    return classi_out[0]['label'], score


def refusal_match(text):
  substrings = ["Sorry" ,"I'm sorry", "I can't", "I am unable to", "I dont know"]
  return any(text.find(substring) != -1 for substring in substrings)


#check for toxicity
def check_toxicity(text):
    detoxify = Detoxify('original')

    # Define a helper function to predict toxicity for a single text string
    def predict_toxicity(text):
        return detoxify.predict(text)

    if text.apply(predict_toxicity)['toxic'] > 1:
      print("Response is toxic")

    else:
      print("Response is not toxic")
      pass
    return text.apply(predict_toxicity)#mock return: {'toxic': 0.002439584396779537,'severe_toxic': 2.191615907795099e-06,'obscene': 0.00010524993886300363,'threat': 1.76735984292135e-06}


#toxicity removing function
def clean_text(text):
    # Load the pre-trained BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the input text
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)

    # Use the BERT model to classify the input text as toxic or non-toxic
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        toxicity_probability = probabilities[:, 1].item()

    # If the input text is classified as toxic (probability > 0.5), remove it
    if toxicity_probability > 0.5:
        return ""

    # Otherwise, return the original text
    return text


def calculate_meteor_score(prompt, response):
    # Convert prompt and response to lowercased strings
    prompt = prompt.lower()
    response = str(response).lower()

    # Tokenize the lowercased strings
    prompt_tokens = prompt.split()
    response_tokens = response.split()

    # Calculate METEOR score
    meteorscore = meteor_score.single_meteor_score(prompt_tokens, response_tokens)

    return meteorscore

def cosine_similarity(prompt, response):
    # Generate the embeddings for the prompt and response
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    prompt_embedding = model.encode([prompt])[0]
    response_embedding = model.encode([response])[0]

    # Calculate the cosine similarity between the embeddings
    cosine_sim = 1 - util.pytorch_cos_sim(torch.tensor([prompt_embedding]), torch.tensor([response_embedding]))[0][0]
    return cosine_sim

def hallucination_detection(prompt, response, meteor_weight=0.7, cosine_weight=0.3):
    meteor_score_val = calculate_meteor_score(prompt, response)
    cosine_sim_val = float(cosine_similarity(prompt, response))

    # Combine scores with specified weights
    hallucination_score = meteor_weight * meteor_score_val + cosine_weight * cosine_sim_val

    return hallucination_score, meteor_score_val, cosine_sim_val

def extract_info(query, answer):
                data = {"prompt": query,
                        "response": answer}
                result = extract(data)
                return result


llm_schema = light_metrics.init()


def extract_info_eng(query, answer):
    data = {"prompt": query,
            "response": answer}
    result_eng = extract(data)
    return result_eng

#Checks for duplication of chunks. (done)
def chunk_checker(top_4_chunks):
        # Check if the top 4 chunks are separate
        print("Checking duplication in chunks...")
        separate_chunks = True
        for i in range(5):
          for j in range(i+1, 5):
            if top_4_chunks[i] == top_4_chunks[j]:
              separate_chunks = False
              break
          if not separate_chunks:
            break
              # Print the result
        if separate_chunks:
            print("The top 5 chunks are separate.")
            #print(chunks_df)
        else:
            print("The top 5 chunks are not separate. The following chunks are duplicates:")
        for i in range(4):
            for j in range(i+1, 4):
                if top_4_chunks[i] == top_4_chunks[j]:
                    print(f"Chunk {i+1}: {top_4_chunks[i]}")
                    print(f"Chunk {j+1}: {top_4_chunks[j]}")
"""
        # Append the separate chunks to the DataFrame
        if separate_chunks:
          for doc, score in top_4_chunks:
            chunks_df = chunks_df.append({'chunk': doc.page_content, 'score': score}, ignore_index=True)
        else:
          for doc, score in top_4_chunks:
            chunks_df = chunks_df.append({'chunk': doc.page_content, 'score': score}, ignore_index=True)
            separate_chunks = True
            for j in range(1, len(top_4_chunks)):
              if doc.page_content == top_4_chunks[j][0].page_content:
                separate_chunks = False
                break
            if separate_chunks:
              break
"""


