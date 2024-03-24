from dotenv import load_dotenv
from transformers import pipeline
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
import os

import requests #For making api requests

# Activating huggingface model
load_dotenv(".env")
KEY = os.getenv('HUGGINGFACEHUB_API_TOKEN')
KEYOPENAI = os.getenv('OPENAI_API_KEY')

# Img2Text convert image to text for the story
def img2text(image):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", max_new_tokens=512)
    text = image_to_text(image)[0]['generated_text']
    print(text)
    return text

# LLM: Generate a short story
def generate_story(scenario):
    template = """
    You are a storyteller. You can generate a short story based on a simple narrative. The story should be no longer than 40 words.
    CONTEXT: {scenario}
    STORY:
    """
    prompt = promo = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm = OpenAI(
        model_name ="gpt-3.5-turbo", temperature = 1), prompt=prompt, verbose = True)
    
    story = story_llm.invoke({"scenario": scenario})
    
    print(story)
    return story["text"]

scenario = img2text("apple.png")
print(scenario)

storyTold = generate_story(scenario)
print(storyTold)

# Text to speech: Make this story into an audio story
def text2speech(message):
    global KEY
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {KEY}"}
    payloads = {
        "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open("audio.flac", "wb") as file:
        file.write(response.content)

scenrario = img2text("apple.png")
story = generate_story(scenario)
text2speech(story)
