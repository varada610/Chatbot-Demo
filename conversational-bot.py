import os
from openai import OpenAI
from together import Together
import tiktoken

# 1. Get the Together AI API Key.
api_key = os.getenv("TOGETHER_API_KEY")

# 2. Declare the global variables needed to interact with the model
client = Together(api_key=api_key)
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TEMPERATURE = 0.8
DEFAULT_BPE = "cl100k_base"
MAX_TOKENS = 100
TOKEN_BUDGET = 250

# Prompt and roles
SYSTEM_PROMPT = "You are a happy obedient assistant who answers questions briefly and politely."
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

# 3. Define the message that contains the system prompt to be sent over to the model
messages = [{"role": SYSTEM, "content": SYSTEM_PROMPT}]

            ##  Encoding and Token mamagement 

# 4. Retrieve the encoding based on the model
def get_encoding(model):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: Tokenizer for model '{model}' not found. Falling back to 'cl100k_base'.")
        return tiktoken.get_encoding(DEFAULT_BPE)

ENCODING = get_encoding(MODEL) 

# 5. Count Tokens
def count_tokens(text):
    return len(ENCODING.encode(text))

# 6. Aggregate total tokens used in the chat session
def total_tokens_used(messages):
    try:
        return sum(count_tokens(msg["content"]) for msg in messages)
    except Exception as e:
        print(f"[token count error]: {e}")
        return 0

# 7. Remove the first token in chat to maintain the history
def enforce_token_budget(messages, budget=TOKEN_BUDGET):
    if total_tokens_used(messages) > budget:
        messages.pop(0)    

def communicate_with_model(messages) :
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    return response.choices[0].message.content
    
# 8. Chat with model passing user input
def chat(user_input):
    messages.append({"role": USER, "content": user_input})
    reply = communicate_with_model(messages)
    messages.append({"role": ASSISTANT, "content": reply})
    enforce_token_budget(messages)  

    return reply

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break
    answer = chat(user_input)
    print("Assistant:", answer)
    print("Current tokens:", total_tokens_used(messages))

