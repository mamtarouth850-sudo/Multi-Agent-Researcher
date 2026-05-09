import glob
import os

agent_files = glob.glob(r'c:\Users\USER\Downloads\multi_agent_researcher\multi_agent_researcher\agents\*.py')
for f in agent_files:
    with open(f, 'r', encoding='utf-8') as file:
        c = file.read()
    
    # Do replacements
    c = c.replace('from langchain_openai import ChatOpenAI', 'from langchain_google_genai import ChatGoogleGenerativeAI')
    c = c.replace('ChatOpenAI(', 'ChatGoogleGenerativeAI(')
    c = c.replace('api_key=config.openai_api_key', 'google_api_key=config.google_api_key')
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(c)
