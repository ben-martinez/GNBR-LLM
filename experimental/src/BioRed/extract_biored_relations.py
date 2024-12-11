import os
from datetime import datetime
import openai
import json
from tqdm import tqdm

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
PROMPTS_DIR = os.path.join(BASE_DIR, '../../resources/prompts/')
OUTPUT_DIR = os.path.join(BASE_DIR, f'../../outputs/biored_outputs/relationships')

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"biored_relations_{timestamp}.json")

input_filename = 'test1'
INPUT_PATH = os.path.join(BASE_DIR, f'../../resources/biored_data/{input_filename}_formatted.json')


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
    abstract_entities = abstract_entry.get('entities', [])
    abstract_id = abstract_entry.get('pbid', '')

    abstract_content = f"{abstract_title}\n\n{abstract_text}"
    abstract_entities_str = json.dumps(abstract_entities, indent=4)


    # ============================
    # Step 1: Gather Relationships
    # ============================
    try:
        gather_relations_prompt = load_prompt('biored_gather_all_relationships.txt')
        additional_inputs = {
            "Abstract" : abstract_content,
            "Canonicalized Entities": abstract_entities_str
        }
        response_text_1 = make_api_call(gather_relations_prompt, additional_inputs=additional_inputs)
        
        relations_json_output = parse_json_response(response_text_1)
        if relations_json_output:
            relations_file_name = f"relationships_{abstract_id}_{timestamp}.json" # TODO change this to not save in individual files
            save_output(relations_json_output, OUTPUT_DIR, relations_file_name, is_json=True)
        else:
            # Save raw response
            save_output(response_text_1, OUTPUT_DIR, f"relationships_{abstract_id}_{timestamp}_raw.txt")
    except Exception as e:
        print(f"Error during Step 1 for {abstract_id}: {e}")
        return


    # ============================
    # Step 2: Reformat Relationships to BioRed Format and types
    # ============================
    try:
        bioredinize_relations_prompt = load_prompt('biored_canonicalize_relationships.txt')
        additional_inputs = {
            "Abstract" : abstract_content,
            "Entities": abstract_entities_str,
            "Relationships": json.dumps(relations_json_output, indent=4)
        }
        response_text_2 = make_api_call(bioredinize_relations_prompt, additional_inputs=additional_inputs)
        
        canonicalized_relations_json_output = parse_json_response(response_text_2)
        if canonicalized_relations_json_output:
            canonicalized_relations_file_name = f"canonicalized_relationships_{abstract_id}_{timestamp}.json" # TODO change this to not save in individual files
            save_output(canonicalized_relations_json_output, OUTPUT_DIR, canonicalized_relations_file_name, is_json=True)
        else:
            # Save raw response
            save_output(response_text_2, OUTPUT_DIR, f"canonicalized_relationships_{abstract_id}_{timestamp}_raw.txt")
    except Exception as e:
        print(f"Error during Step 2 for {abstract_id}: {e}")
        return
    
    # save the abstract data including the id, abstract title, text, entities, and canonicalized relations to the output JSON file by appending the abstract data as a dictionary to the list in the JSON file
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    
    abstract_data = {
        "id": abstract_id,
        "title": abstract_title,
        "text": abstract_text,
        "entities": abstract_entities,
        "relationships": canonicalized_relations_json_output
    }

    output_data.append(abstract_data)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    
    
    print(f"Processing of {abstract_id} completed.\n")


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
