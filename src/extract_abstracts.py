import logging
import os
import glob
from datetime import datetime
import openai
import json
from utils import extract_title_and_abstract

# Input OpenAI API Key
API_KEY = ""

client = openai.OpenAI(
    api_key=API_KEY,
)

# Paths to the prompt, abstracts, and output directory
prompt_path = os.path.join(os.path.dirname(__file__), '../resources/prompts/prompt.txt')
abstracts_path = os.path.join(os.path.dirname(__file__), '../resources/abstracts/*.txt')
output_dir = os.path.join(os.path.dirname(__file__), '../outputs/')

os.makedirs(output_dir, exist_ok=True)

# Get prompt
with open(prompt_path, 'r') as file:
    prompt = file.read()

# Get abstracts
abstracts = ''
for abstract_file in glob.glob(abstracts_path):
    with open(abstract_file, 'r') as file:
        abstracts += file.read() + '\n'

full_prompt = f"{prompt}\n\n{abstracts}"

# Prepare the messages for the chat completion
messages = [
    {"role": "user", "content": full_prompt}
]

# Make the API call to the chat completion endpoint
llm_name = "gpt-4o-mini"

response = client.chat.completions.create(
    model=llm_name,
    messages=messages,
    temperature=0.7
)

# Extract the response text and parse as JSON
response_text = response.choices[0].message.content

print(response_text)

response_text = str(response_text).strip()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f"output_{timestamp}.txt"
output_filepath = os.path.join(output_dir, output_filename)

# Save the response to the output file
with open(output_filepath, 'w') as file:
    file.write(response_text)

print(f"Response saved to {output_filepath}")

# Save the response JSON
output_filename = f"output_{timestamp}.json"
output_filepath = os.path.join(output_dir, output_filename)

# Attempt to parse and save JSON output
try:
    # Split response text into lines and strip any Markdown-style code block markers
    lines = response_text.splitlines()
    
    # Remove ```json from the first line and ``` from the last line
    if lines[0].startswith("```json"):
        lines = lines[1:]  # Remove the first line
    if lines[-1].strip() == "```":
        lines = lines[:-1]  # Remove the last line
    
    # Join the remaining lines back into a string
    json_string = "\n".join(lines).strip()
    
    # Attempt to parse the cleaned string as JSON
    json_output = json.loads(json_string)
    output_filepath = os.path.join(output_dir, f"output_{timestamp}.json")
    with open(output_filepath, 'w', encoding='utf-8') as file:
        json.dump(json_output, file, indent=4)
    logging.info(f"Response JSON saved to {output_filepath}")
except (json.JSONDecodeError, ValueError) as e:
    logging.warning(f"JSON parsing failed: {e}. Saving raw response.")
    output_filepath = os.path.join(output_dir, f"output_{timestamp}_raw.txt")
    with open(output_filepath, 'w', encoding='utf-8') as file:
        file.write(response_text)
    logging.info(f"Raw response saved to {output_filepath} (not JSON formatted).")
