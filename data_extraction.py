import fitz  # PyMuPDF
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
import os
from pathlib import Path
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, TapasTokenizer, TapasForQuestionAnswering
import openai
import os
import logging
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Initialize Pegasus model and tokenizer
#pegasus_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
#pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

# Initialize TAPAS model and tokenizer
tapas_model = TapasForQuestionAnswering.from_pretrained('google/tapas-large-finetuned-wtq')
tapas_tokenizer = TapasTokenizer.from_pretrained('google/tapas-large-finetuned-wtq')

model = SentenceTransformer('bert-base-nli-mean-tokens')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_and_images(pdf_path):
    """
    Extract text and images from a PDF file.
    pdf_path: Path to the PDF file
    """
    doc = fitz.open(pdf_path)
    filename = Path(pdf_path).stem
    texts = []
    images = []
    text_metadata = []
    image_metadata = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        texts.append(page_text)
        text_metadata.append({
            "source_file": pdf_path,
            "page_number": page_num + 1,
            "chunk_size": len(page_text)
        })
        
        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_name = f"image_page{page_num+1}_{img_index}_{filename}.{image_ext}"
            images.append((image_name, image_bytes, page_num + 1))
            image_metadata.append({
                "source_file": pdf_path,
                "page_number": page_num + 1,
                "image_name": image_name
            })
            
            # Optionally, save the image
            with open(os.path.join("static", image_name), "wb") as image_file:
                image_file.write(image_bytes)
    
    return texts, images, text_metadata, image_metadata

def extract_tables(pdf_path):
    """
    Extract tables from a PDF file.
    pdf_path: Path to the PDF file
    """
    tables = []
    table_metadata = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                tables.append(table)
                table_metadata.append({
                    "source_file": pdf_path,
                    "page_number": page_num + 1,
                    "table_size": len(table)
                })
    
    return tables, table_metadata

def categorize_elements(texts, tables, images):
    """
    Categorize extracted elements from a PDF into texts, tables, and images.
    """
    return texts, tables, images

def process_pdfs_in_directory_uploadpdf(directory_path):
    """
    Process all PDF files in the specified directory.
    directory_path: Path to the directory containing PDF files
    """
    # Ensure 'static' directory exists to save images
    os.makedirs("uploaded", exist_ok=True)

    pdf_data = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")

            # Extract data from the PDF
            raw_texts, raw_images, text_metadata, image_metadata = extract_text_and_images(pdf_path)
            raw_tables, table_metadata = extract_tables(pdf_path)
            texts, tables, images = categorize_elements(raw_texts, raw_tables, raw_images)



            # Store the extracted data in a dictionary
            pdf_data.append({
                "filename": filename,
                "texts": texts,
                "tables": tables,
                "images": [image[0] for image in images],
                "text_metadata": text_metadata,
                "table_metadata": table_metadata,
                "image_metadata": image_metadata
            })

    for data in pdf_data:
        print(f"Filename: {data['filename']}")
        print("Texts:", data['texts'])
        print("Text Metadata:", data['text_metadata'])
        print("Tables:", data['tables'])
        print("Table Metadata:", data['table_metadata'])
        print("Images:", data['images'])
        print("Image Metadata:", data['image_metadata'])

    return pdf_data, texts, tables


def text_chunks_list(texts):
    text_chunk_list_uploadpdf = []
    for i, chunk in enumerate(texts):
        text_chunk_list_uploadpdf.append(chunk)

    return text_chunk_list_uploadpdf



# Function to generate questions and answers from tables using TAPAS
def generate_table_qa_with_tapas(tables):
    """
    Generate Q&A from table elements using TAPAS
    tables: List of lists (tables in nested list format)
    """
    def generate_qa(table):
        try:
            # Convert the nested list to a pandas dataframe and fill None values with empty strings
            df = pd.DataFrame(table[1:], columns=table[0]).fillna('')
            logger.info(f"DataFrame:\n{df}")

            # List of questions to generate insights from the table
            questions = [
                "What is the summary of this table?",
                "What are the key findings?",
                "What are the highest values in each category?",
                "What are the lowest values in each category?"
            ]

            qa_results = []
            for question in questions:
                inputs = tapas_tokenizer(table=df, queries=[question], padding='max_length', return_tensors="pt")
                logger.info(f"Inputs: {inputs}")

                outputs = tapas_model(**inputs)
                logger.info(f"Outputs: {outputs}")

                predicted_answer_coordinates, predicted_aggregation_indices = tapas_tokenizer.convert_logits_to_predictions(
                    inputs,
                    outputs.logits.detach(),
                    outputs.logits_aggregation.detach()
                )
                logger.info(f"Predicted Coordinates: {predicted_answer_coordinates}")

                # Extract the predicted answer text
                answers = []
                for coordinates in predicted_answer_coordinates[0]:
                    if coordinates:
                        cell_value = df.iat[coordinates[0], coordinates[1]]
                        answers.append(cell_value)
                
                qa_results.append(f"Q: {question} A: {'; '.join(answers)}")

            return "\n".join(qa_results)
        except Exception as e:
            logger.error(f"Error generating Q&A from table: {e}")
            return "Error generating Q&A from table."

    return [generate_qa(table) for table in tables]

# Function to generate summaries
def generate_table_summaries(tables=None, summarize_texts_flag=False):
    """
    Summarize text and table elements
    text_chunks: List of text chunks
    tables: List of lists (tables in nested list format)
    summarize_texts_flag: Bool to summarize texts
    """
    table_summaries = []


    # Generate Q&A from tables
    if tables:
        table_summaries = generate_table_qa_with_tapas(tables)

    return table_summaries

def get_embedding(text):
    try:
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding for text: {text}. Error: {e}")
        return None


def upload_texts_to_vectorstore(text_chunk_list, data, index_name="pdfupload"):
    pc = Pinecone(api_key="6ab49224-b6f2-4358-8430-28b8af6fcde3")

    # Check if the index exists
    if index_name in pc.list_indexes().names():
        index = pc.Index(index_name)
        
        # Check if the index is empty
        stats = index.describe_index_stats()
        if stats['total_vector_count'] > 0:
            # Wipe the index
            logger.info(f"Index '{index_name}' is not empty. Deleting existing vectors.")
            index.delete(delete_all=True)
        else:
            logger.info(f"Index '{index_name}' is empty.")
    else:
        # Create a new index
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



    text_embedding = [get_embedding(i) for i in text_chunk_list]

    text_metadata_lst = [
        data['text_metadata'][i] for i, text in enumerate(data['texts'])
    ]

    for i, embedding in enumerate(text_embedding):
        if embedding is not None:
            try:
                response = index.upsert(
                    vectors=[
                        {
                            'id': f"text_{i}",
                            'values': embedding.tolist(),
                            'metadata': {**text_metadata_lst[i], 'original_text': text_chunk_list[i]}  # Add original text to metadata
                        }
                    ]
                )
                #logger.info(f"Upsert response: {response}")
            except Exception as e:
                logger.error(f"Error upserting text data for text_{i}: {e}")
        else:
            logger.warning(f"Embedding for text_{i} is None")




# Example usage:
# upload_texts_to_vectorstore(text_chunk_list, data)


# Example usage:
# upload_texts_to_vectorstore(text_chunk_list, data)


        



# Directory path
#directory_path = "/Users/dhvani/Desktop/blossombot/"

# Process all PDFs in the directory
#pdf_data, texts, tables = process_pdfs_in_directory(directory_path)





