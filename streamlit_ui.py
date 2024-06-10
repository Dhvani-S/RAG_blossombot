import streamlit as st
import openai
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import logging
from openai import OpenAI
import responsible_tests as rt
from detoxify import Detoxify
import time
import PyPDF2
import pandas as pd
import mongo_connect as mc
import tempfile
import data_extraction as de
from concurrent.futures import ThreadPoolExecutor
import base64
from dotenv import load_dotenv

load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
index_name_upload = os.getenv("PINECONE_INDEX_NAME_UPLOAD")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=pinecone_api_key, index_name=index_name, index_name_upload=index_name_upload)
# Initialize the OpenAI client


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


detoxify = Detoxify('original')

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)
index_upload = pc.Index(index_name_upload)

df = pd.DataFrame()

# Initialize session state if not already done
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Initialize the SentenceTransformer model
model = SentenceTransformer('bert-base-nli-mean-tokens')
session_id = 'sess' + str(time.time())
query_counter = 0

# Display the logo in the sidebar
logo_path = "/Users/dhvani/Downloads/Untitled_Artwork 2-2.png"  # Replace with your logo file path
with open(logo_path, "rb") as image_file:
    logo_base64 = base64.b64encode(image_file.read()).decode()

# Background image path
background_image_path = "/Users/dhvani/Desktop/blossombot/bg.jpeg"  # Replace with your background image file path

# Convert the background image to base64
with open(background_image_path, "rb") as image_file:
    background_image_base64 = base64.b64encode(image_file.read()).decode()

# Custom CSS for background image and sidebar
st.markdown(
    f"""
    <style>
    .navbar {{
        width: 100%;
        height: 100px;
        position: fixed;
        top: 3;
        left: 0;
        background: transparent;
        display: flex;
        align-items: center;
        padding: 0 20px;
        z-index: 1000;
    }}
    .navbar img {{
        height: 70px;
    }}
    .stApp {{
        background: url("data:image/png;base64,{background_image_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .chat-history {{
        max-height: 300px;
        overflow-y: scroll;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    .user-msg {{
        text-align: right;
        background-color: #A37AA1;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        max-width: 60%;
        float: right
    }}
    .assistant-msg {{
        text-align: left;
        background-color: #789C96;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        max-width: 60%;
    }}
    .stChatInput {{
        background-color: transparent !important;
        color: #000 !important;
        border: 2px solid #ccc !important;
        border-radius: 12px !important;
        border-color: #0C5C99;
        padding: 12px !important;
        margin: 10px 0 !important;
    }}
    .stChatInput > div > textarea {{
        background-color: transparent !important;
        color: #333 !important;
        font-size: 16px !important;
        border: 5px !important;
        border-color: #0C5C99;
        border-radius: 10px !important;
        padding: 10px !important;
    }}
    .stSpinner > div > div {{
        color: #0C5C99 !important;  /* Change spinner font color here */
        font-size: 20px !important; /* Adjust font size if needed */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div class="navbar">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Add file uploader to sidebar
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

def display_conversation():
    st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
    for msg in st.session_state.conversation:
        role = "user-msg" if msg["role"] == "user" else "assistant-msg"
        st.markdown(f"""
            <div class='{role}'>
                <strong>{msg['role'].capitalize()}:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Using st.chat_input
query = st.chat_input("What do you want to know?", key="query_input")

def process_query(query):
    start_time = time.time()
    st.session_state.conversation.append({"role": "user", "content": query})

    query_vector = model.encode(query).tolist()

    def query_pinecone(index, query_vector):
        return index.query(vector=query_vector, top_k=5, namespace='', include_values=True, include_metadata=True)

    with ThreadPoolExecutor() as executor:
        future_search_response = executor.submit(query_pinecone, index, query_vector)
        future_search_response_upload = executor.submit(query_pinecone, index_upload, query_vector)

        search_response = future_search_response.result()
        search_response_upload = future_search_response_upload.result()

    chunks, chunks_lst, chunk_id_lst = [], [], []
    
    for i, item in enumerate(search_response['matches']):
        original_t = item['metadata'].get('original_text')
        chunks_lst.append(original_t)
        chunk_id_lst.append(item['id'])
        chunks.append(f"Chunk {i} - {item['id']} (Score: {item['score']}):\nOriginal Text: {original_t}\nMetadata: {item['metadata']}")

    df['chunks'] = chunk_id_lst
    df['query'] = query
    df['query_toxic_score'] = detoxify.predict(query)['toxicity']
    df['start_time'] = start_time
    joined_chunks = "\n\n".join(chunks)
    rt.chunk_checker(chunks_lst)

    chunks_upload, chunks_lst_upload, chunk_id_lst_upload = [], [], []
    for j, item1 in enumerate(search_response_upload['matches']):
        original_t_upload = item1['metadata'].get('original_text')
        chunks_lst_upload.append(original_t_upload)
        chunk_id_lst_upload.append(item1['id'])
        chunks_upload.append(f"Chunk {j} - {item1['id']} (Score: {item1['score']}):\nOriginal Text: {original_t_upload}\nMetadata: {item1['metadata']}")

    joined_chunks_upload = "\n\n".join(chunks_upload)

    # Initialize or update the conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Add the user's query to the conversation history
    st.session_state.conversation_history.append({"role": "user", "content": query})

    # Determine the context to use based on available chunks
    context_chunks = joined_chunks if not joined_chunks_upload else joined_chunks_upload

    # Create the prompt including the context
    prompt = f"""Role: Greet yourself as BlossomBot if you haven't before. Only greet once. You are a bot who talks about women's health based on uploaded documents. You have to answer any query related to women's health and guide the user like the experienced and professional gynecologist. If someone discusses symptoms of certain disease, you have to act like a doctor and engage them in conversation before jumping to any conclusion. Try to quote the reference in answers too.
    Answer the following question based on the context below.
    If you don't know the answer, search for an answer from verified medical resources and give appropriate references.
    ---
    CONTEXT:
    {context_chunks}
    """

    # Append the conversation history to the prompt
    for message in st.session_state.conversation_history:
        prompt += f"{message['role'].upper()}: {message['content']}\n"

    # Generate the response using the chatbot model
    with st.spinner("Generating response..."):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

        answer = response.choices[0].message.content
        df['response'] = answer

        bias_class, bias_score = rt.debiasing(answer)
        df['bias_class'] = bias_class
        df['bias_score'] = bias_score

        eng_result = rt.extract_info_eng(query, answer)
        df1 = pd.DataFrame(eng_result.items())
        transposed_df = df1.T
        df1.reset_index(drop=True, inplace=True)

        refusal = 'Detected' if rt.refusal_match(answer) else 'Not Detected'
        df['refusal'] = refusal

        answer_toxic_score = detoxify.predict(answer)['toxicity']
        df['response_toxic_score'] = answer_toxic_score

        hallucination_score, meteor_score_val, cosine_sim_val = rt.hallucination_detection(query, answer, meteor_weight=0.7, cosine_weight=0.3)
        df['halucination_score'] = hallucination_score
        df['meteor_score'] = meteor_score_val
        df['cosine_sim_score'] = cosine_sim_val

        st.session_state.conversation.append({"role": "assistant", "content": answer})

        time_taken = time.time() - start_time
        df['time_taken'] = time_taken

        combined_row = {
            'chunks': df['chunks'].tolist(),
            'query': df['query'].iloc[0],
            'query_toxic_score': df['query_toxic_score'].iloc[0],
            'start_time': df['start_time'].iloc[0],
            'halucination_score': df['halucination_score'].iloc[0],
            'meteor_score': df['meteor_score'].iloc[0],
            'cosine_sim_score': df['cosine_sim_score'].iloc[0],
            'time_taken': df['time_taken'].iloc[0]
        }

        df_combined = pd.DataFrame([combined_row])
        df_combined['refusal'] = refusal
        df_combined['session_id'] = session_id
        df_combined['response'] = answer
        df_combined['response_toxic_score'] = answer_toxic_score
        df_combined['bias_class'] = bias_class
        df_combined['bias_score'] = bias_score

        df_combined.to_csv("result.csv")

        collection = mc.mongo_connect()
        data_dict = df_combined.to_dict("records")
        collection.insert_many(data_dict)

if query:
    process_query(query)

if uploaded_file is not None:
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name
    
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())

    pdf_data, texts, tables = de.process_pdfs_in_directory_uploadpdf(temp_dir_path)

    for data in pdf_data:
        print(f"Filename: {data['filename']}")
        print("Texts:", data['texts'])
        print("Text Metadata:", data['text_metadata'])
        print("Tables:", data['tables'])
        print("Table Metadata:", data['table_metadata'])
        print("Images:", data['images'])
        print("Image Metadata:", data['image_metadata'])

    texts_chunk = de.text_chunks_list(texts)
    upload_table_summaries = de.generate_table_summaries(tables)

    de.upload_texts_to_vectorstore(texts_chunk, data)
    st.session_state.temp_dir = temp_dir

# Display the conversation history
display_conversation()
