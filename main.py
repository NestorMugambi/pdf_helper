from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
load_dotenv()


st.set_page_config(
    page_title="pdf_assistant",
    page_icon="📘"
)
with st.sidebar:
        st.title('🤗💬 LLM Chat App')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
                    
        
        ''')
        


load_dotenv()
def main():
    st.header("Chat with your pdf📃📃")
    
    pdf = st.file_uploader("Upload your pdf file" , type ="pdf")  
       
             
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text =""

        for page in pdf_reader.pages:
            text+=page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text = text)
        #embeddings
       
        store_name = pdf.name[:-4]

        
        if os.path.exists(f"{store_name}"):
            VectorStore = FAISS.load_local(f"{store_name}", OpenAIEmbeddings())

        else:            
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
            
            VectorStore.save_local(f"{store_name}")       
    

    if "messages" not in st.session_state:
       st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])    

    query = st.chat_input("Ask questions about your pdf")    

    if query:
        if  pdf:           
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)
                
                docs = VectorStore.similarity_search(query=query)
                llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                                convert_system_message_to_human=True,
                                                safety_settings={
                                                
                                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
                                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                                },
                    )      
                    
                chain =load_qa_chain(llm=llm,chain_type="stuff")
                response = chain.run(input_documents=docs,question=query)
                        
                with st.chat_message("assistant"):
                    st.markdown(response)

                    st.session_state.messages.append({"role": "assistant", "content": response})  
        else:
             st.warning("Upload a pdf file to continue")     
        
    else:        
            ask = st.chat_input("Ask general questions ",key ="ask")
            if ask:
                st.session_state.messages.append({"role": "user", "content": ask})
                with st.chat_message("user"):
                    st.markdown(ask)            
                
                llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                            convert_system_message_to_human=True,
                                            safety_settings={
                                            
                                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
                                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                            },
                )
                prompt_template_name=PromptTemplate(
                                    input_variables=["question"],
                                    template="""
                                            answer only the question asked in {question} and format it in a way easy to understand        
                                            """
                                )
                prompt1=prompt_template_name.format(question=ask)
                    
                
                result = llm.invoke(prompt1)            
                        
                with st.chat_message("assistant"):
                    st.markdown(result.content)

                    st.session_state.messages.append({"role": "assistant", "content": result.content})
                
                         
      
if __name__ == "__main__":
     
     main()
    

   

        

