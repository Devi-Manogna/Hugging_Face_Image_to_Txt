from dotenv import find_dotenv, load_dotenv
import os
import tensorflow as tf
#library used to create a user interface for our python code
import streamlit as st 

import openai
from openai import OpenAI

tf.config.experimental_run_functions_eagerly(True)  # This will relax the shape requirements


#pipeline allows to download huggingface mdoel into the local machine
from transformers import pipeline 
#from langchain import PromptTemplate, LLMChain, OpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI


#to be able to access the hugging face api token
load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
from langchain_community.llms import OpenAI


import requests

#image to text model
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)
    print(text)
    return text

img2text("image3.jpeg")

#llm (to generate short story)
def generate_story(scenario):
    template = """
    You are a story teller;
    you can generate a short story based on the simple narrative, this story should be no more than 20 words;
    
    CONTEXT : {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template,input_variables = ["scenario"])
    story_llm = LLMChain(llm=OpenAI(model_name='gpt-3.5-turbo', temperature=1,openai_api_key='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',max_length=20), prompt=prompt, verbose=True)


    story=story_llm.predict(scenario=scenario)
    print(story)
    return story

def main():
    st.set_page_config(page_title='image to text story')
    st.header('Turn image to text story')
    uploaded_file = st.file_uploader("choose an image..",type='jpeg')

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name,'wb') as file:
            file.write(bytes_data)
        st.image(uploaded_file,caption='uploaded image',use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story=generate_story(scenario)

if __name__ == '__main__':
    main()