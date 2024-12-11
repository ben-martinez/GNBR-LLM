import os
from datetime import datetime
import openai
import json
from tqdm import tqdm
import re
import xml.etree.ElementTree as ET

# ============================
# Configuration and Setup
# ============================

# Input OpenAI API Key
API_KEY = ""
CLIENT = openai.OpenAI(
    api_key=API_KEY,
)

# Set up the OpenAI client
openai.api_key = API_KEY

# Paths to the prompts, abstracts, and output directory
BASE_DIR = os.path.dirname(__file__)
input_filename = 'full_test'
INPUT_PATH = os.path.join(BASE_DIR, f'../../resources/biored_data/{input_filename}_formatted.json')

PROMPTS_DIR = os.path.join(BASE_DIR, '../../resources/prompts/')
OUTPUT_DIR = os.path.join(BASE_DIR, f'../../outputs/biored_outputs/ents')

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"biored_tagged_ents_{timestamp}_{input_filename}.json")



# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model name
MODEL_NAME = "gpt-4o-mini"  # Update with your actual model name

# ============================
# Helper Functions
# ============================

def load_prompt(prompt_filename):
    prompt_path = os.path.join(PROMPTS_DIR, prompt_filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt = file.read().strip()
        return prompt
    except FileNotFoundError:
        print(f"Prompt file '{prompt_filename}' not found in {PROMPTS_DIR}.")
        raise

def make_api_call(prompt, additional_inputs=None, temperature=0.7):
    """
    Make an API call to OpenAI's ChatCompletion endpoint.
    
    Args:
        prompt (str): The prompt to send to the model.
        additional_inputs (dict, optional): Additional inputs to include in the prompt.
        temperature (float, optional): Sampling temperature.
    
    Returns:
        str: The raw response text from the API.
    """
    full_prompt = prompt
    if additional_inputs:
        for key, value in additional_inputs.items():
            full_prompt += f"\n\n{key}:\n{value}"
    
    messages = [
        {"role": "user", "content": full_prompt}
    ]
    
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature
        )
        response_text = response.choices[0].message.content.strip()
        #print(response_text)
        return response_text
    except Exception as e:
        print(f"API call failed: {e}")
        raise

def parse_json_response(response_text):
    """
    Parse JSON from the API response, removing Markdown code block markers if present.
    
    Args:
        response_text (str): The raw response text from the API.
    
    Returns:
        dict: The parsed JSON object.
    """
    lines = response_text.splitlines()
    
    # Remove ```json from the first line and ``` from the last line if present
    if lines and lines[0].startswith("```json"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    
    json_string = "\n".join(lines).strip()
    
    try:
        json_output = json.loads(json_string)
        return json_output
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}. Saving raw response.")
        return None

def save_output(content, output_dir, filename, is_json=False):
    output_path = os.path.join(output_dir, filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            if is_json:
                json.dump(content, file, indent=4)
            else:
                file.write(content)
        print(f"Response saved to {output_path}")
    except Exception as e:
        print(f"Failed to save {filename}: {e}")

# ============================
# Main Processing Function
# ============================

def process_abstract(abstract_entry):

    abstract_title = abstract_entry.get('title', '')
    abstract_text = abstract_entry.get('abstract_text', '')
    abstract_id = abstract_entry.get('pbid', '')

    abstract_content = f"{abstract_title}\n\n{abstract_text}"


    # ============================
    # Step 1: Tag ents in the abstract
    # ============================
    try:
        tag_ents_prompt = load_prompt('biored_tag_all_entities.txt')
        additional_inputs = {
            "Abstract" : abstract_content,
        }
        response_text_1 = make_api_call(tag_ents_prompt, additional_inputs=additional_inputs)
        
        # Save raw response
        save_output(response_text_1, OUTPUT_DIR, f"tagged_ent_{abstract_id}_{timestamp}.txt")
    except Exception as e:
        print(f"Error during Step 1 for {abstract_id}: {e}")
        return


    new_json_entry = reformat_response_to_json(response_text_1, abstract_entry)

    # save the new_json_entry to the output file
    append_and_save_output(new_json_entry, OUTPUT_PATH)

def append_and_save_output(entry, output_path):
    # should append the entry to the JSON file at the path
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data.append(entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def reformat_response_to_json(response_text, abstract_entry):
    """
    Reformat the tagged XML-style text into JSON format, extracting entities along with their type and name.
    
    Args:
        response_text (str): The XML-style tagged text output from the model.
        abstract_entry (dict): Dictionary containing the `pbid`, `title`, and `abstract_text`.
    
    Returns:
        dict: A JSON object containing the pbid, title, abstract_text, tagged_abstract_text, and extracted entities.
    """
    # Parse the XML-like tags in the response
    entities = []
    try:
        # Extract all tags using regex to handle non-standard XML strings
        tag_pattern = re.compile(r'<info type="([^"]+)">(.*?)</info>')
        matches = tag_pattern.findall(response_text)
        
        # Populate entities list with unique entries
        seen = set()
        for entity_type, entity_name in matches:
            entity_name = entity_name.strip()
            if (entity_name, entity_type) not in seen:
                entities.append({"name": entity_name, "type": entity_type})
                seen.add((entity_name, entity_type))
    except Exception as e:
        print(f"Error parsing response_text: {e}")
        return {}

    # Construct the final JSON object
    result = {
        "pbid": abstract_entry.get("pbid"),
        "title": abstract_entry.get("title"),
        "abstract_text": abstract_entry.get("abstract_text"),
        "tagged_abstract_text": response_text,
        "entities": entities
    }
    
    return result


def create_output_file(output_path):
    # should create a JSON file at the path with an empty list
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([], f)


# ============================
# Main Execution
# ============================

def main():
    
    create_output_file(OUTPUT_PATH)

    # input path has a JSON file where each entry is a dictionary with the keys 'title' and 'abstract', we want to iterate over all entries
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        abstract_data = json.load(f)
    
    for entry in tqdm(abstract_data):
        process_abstract(entry)

    print("All processing completed.")

if __name__ == "__main__":
    main()
