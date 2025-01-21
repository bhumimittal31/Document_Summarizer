import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader
import fitz
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import pytesseract
from PIL import Image
import torch
import base64
import os
from dotenv import load_dotenv

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Add your Hugging Face token
#token = HUGGING_FACE_TOKEN

# Model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint, token=HUGGING_FACE_TOKEN)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, torch_dtype=torch.float32, token=HUGGING_FACE_TOKEN
)

# File loader and preprocessing
# def file_preprocessing(file_path):
#     file_extension = os.path.splitext(file_path)[-1].lower()

#     # If the file is a PDF, use PyPDFLoader
#     if file_extension == ".pdf":
#         loader = PyPDFLoader(file_path)
#         pages = loader.load_and_split()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#         texts = text_splitter.split_documents(pages)
#         final_texts = "".join(text.page_content for text in texts)
#         return final_texts

#     # If the file is an image, use OCR
#     elif file_extension in [".jpg", ".jpeg", ".png"]:
#         image = Image.open(file_path)
#         ocr_text = pytesseract.image_to_string(image)
#         return ocr_text

#     else:
#         st.error("Unsupported file format!")
#         return None

def file_preprocessing(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()

    # If the file is a PDF, use PyMuPDF
    if file_extension == ".pdf":
        text = ""
        try:
            with fitz.open(file_path) as pdf_doc:
                for page in pdf_doc:
                    text += page.get_text()
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None
        return text

    # If the file is an image, use OCR
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        try:
            image = Image.open(file_path)
            ocr_text = pytesseract.image_to_string(image)
        except Exception as e:
            st.error(f"Error reading image: {str(e)}")
            return None
        return ocr_text

    else:
        st.error("Unsupported file format!")
        return None
# LLM pipeline for Summarization and Key Points Extraction
# LLM pipeline for Summarization and Key Points Extraction
def llm_pipeline(filepath, summary_length):
    # Define length parameters for each option
    length_params = {
        "Small": {"max_length": 150, "min_length": 30},
        "Medium": {"max_length": 300, "min_length": 50},
        "Large": {"max_length": 500, "min_length": 100},
    }

    # Get length settings based on user selection
    max_len = length_params[summary_length]["max_length"]
    min_len = length_params[summary_length]["min_length"]

    # Initialize summarization pipeline
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=max_len,
        min_length=min_len
    )
    
    input_text = file_preprocessing(filepath)
    if not input_text:
        return None, None

    # Summarize the text
    summary = pipe_sum(input_text)
    summary_text = summary[0]['summary_text']
    
    # Extract Key Points (using sentence tokenization and frequency analysis)
    sentences = input_text.split('.')
    key_points = sorted(sentences, key=lambda x: len(x.split()), reverse=True)[:5]  # Pick top 5 longest sentences as key points
    
    return summary_text, key_points


@st.cache_data
# Function to display the PDF or Image
def display_file(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()
    
    if file_extension == ".pdf":
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        st.image(file_path, caption="Uploaded Image", use_container_width=True)

# Streamlit Code
st.set_page_config(layout="wide")

# Streamlit Code
def main():
    st.title("Document and Image Summarization App")

    # Upload file
    uploaded_file = st.file_uploader("Upload your file (PDF or Image)", type=['pdf', 'jpg', 'jpeg', 'png'])

    # Add summary length selection
    summary_length = st.radio(
        "Select Summary Length",
        options=["Small", "Medium", "Large"],
        index=1  # Default to "Medium"
    )

    if uploaded_file is not None:
        filepath = f"data/{uploaded_file.name}"
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        
        if st.button("Summarize"):
            col1, col2 = st.columns(2)

            with col1:
                st.info("Uploaded File")
                display_file(filepath)

            with col2:
                summary, key_points = llm_pipeline(filepath, summary_length)
                if summary:
                    st.info("Summarization Complete")
                    st.success(summary)
                    st.info("Key Points:")
                    for point in key_points:
                        st.write(f"- {point.strip()}")
                else:
                    st.error("Error in processing the file!")

if __name__ == "__main__":
    main()
