from pypdf import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR
import os
from dotenv import load_dotenv
import pickle
#from transformers import AutoModel, AutoTokenizer
from langchain.llms.google_palm import GooglePalm
from langchain.chains.question_answering import load_qa_chain

with st.sidebar:
    st.title('LLM Chat App 🥳')
    st.markdown('''
    this model is created by using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [GooglePalm](https://ai.google.dev/palm_docs#:~:text=The%20PaLM%20API%20is%20the,the%20latest%20API%20and%20models.) LLM model''')
    
load_dotenv()
def main():
    st.header('PDF Dialogue Zone 📚💬')
    p=st.file_uploader('Upload your document',type='pdf')
    if p is not None:
        pdf=PdfReader(p)
        text=""
        for page in pdf.pages:
            text += page.extract_text()
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)
        #st.write(chunks)

        
        name=p.name[:-4]
        #st.write(name)

        if os.path.exists(f"{name}.pkl"):
            with open(f"{name}.pkl","rb") as f:
                vector=pickle.load(f)
            #st.write('embeddings from disk')
        else:
            embeddings = HuggingFaceInstructEmbeddings()
            vector=FAISS.from_texts(chunks, embedding=embeddings)
            with open (f"{name}.pkl","wb") as f:
                pickle.dump(vector,f)
            #st.write('embedings done')
        #ask que
        query=st.text_input('Ask Questions About Your PDF File:')
        if query:

            doc=vector.similarity_search(query=query,k=2)
            #st.write(doc)
            
            llm=GooglePalm()
            chain=load_qa_chain(llm=llm,chain_type='stuff')
            response=chain.run(input_documents=doc, question=query)
            st.write(response)
            #
            
        
if __name__=='__main__':
    main()
