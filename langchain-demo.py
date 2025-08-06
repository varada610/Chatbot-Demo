import os
from langchain_together import Together
from langchain.prompts import PromptTemplate

# 1. Get the Together AI API Key.
API_KEY = os.getenv("TOGETHER_API_KEY")

# 2. Declare the global variables needed to interact with the model
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
TEMPERATURE = 0.8

# 3. Initialize the model with the model global model related variables
llm = Together(model=MODEL, 
               temperature=TEMPERATURE, 
               max_tokens=250,
               together_api_key=API_KEY)

# 4. Template with placeholder for text , placeholder varible to be populated
template = "Translate the text to french please : {text}."
prompt_template = PromptTemplate.from_template(template)

# 5. Pipe the llm model to the prompt template to create a chain
chain = prompt_template.pipe(llm)

# 6. Format the the prompt template to see rendered text sent over to the model
prompt_string = prompt_template.format(text="Hello! how are you.")
print(prompt_string)

# 7. The placeholder text is translated based on the template and model
response = chain.invoke({"text": "Hello! how are you."})
print(response) 