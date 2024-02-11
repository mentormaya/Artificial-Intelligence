import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# MODEL AND TOKENIZER
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map = "auto", torch_dtype = torch.float32)

#File loader and processing
def file_processing(file):
  loader = PyPDFLoader(file)
  pages = loader.load_and_split()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
  texts = text_splitter.split_documents(pages)
  final_texts = ""
  for text in texts:
    print(text)
    final_texts += text.page_content
  return final_texts

#LM Pipeline
def llm_pipeline(filepath):
  pipe_sum = pipeline(
    "summarization",
    model = base_model,
    tokenizer=tokenizer,
    max_length = 500,
    min_length = 50
  )
  input_text = file_processing(filepath)
  result = pipe_sum(input_text)
  result = result[0]["summary_text"]
  return result

#Streamlit codes

@st.cache_data
# function to display the given PDF file
def displayPDF(file):
  # opening file from the filepath
  with open(file, "rb") as pdf:
    base64PDF = base64.b64encode(pdf.read()).decode("utf-8")
  
  # Embedding PDF in HTML
  pdf_display = f'<iframe src="data:application/pdf;base64;{base64PDF}" width="100%" height="600" type="application/pdf"></iframe>'

  # Displaying the file in to streamlit UI
  st.markdown(pdf_display, unsafe_allow_html=True)

# setting wide screen for the UI
st.set_page_config(layout="wide", page_title="Summarization App")

def main():
  st.title("PDF summarization App with AI and LLM")
  uploaded_file = st.file_uploader("Upload Your PDF file", type=["pdf"])
  if uploaded_file is not None:
    if st.button("Summarize"):
      original, result = st.columns(2)
      filepath = f"uploads/{uploaded_file.name}"
      with open(filepath, "wb") as tempFile:
        tempFile.write(uploaded_file.read())
      with original:
        st.info("Uploaded File here!")
        pdfViewer = displayPDF(filepath)
      
      with result:
        st.info("Summarized result will be here!")
        summary = llm_pipeline(filepath)
        st.success(summary)

if __name__ == "__main__":
  main()