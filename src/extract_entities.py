import os
import glob
from datetime import datetime
import openai
import json
from utils import extract_title_and_abstract
import graph_utils
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
PROMPTS_DIR = os.path.join(BASE_DIR, '../resources/prompts/')
ABSTRACTS_DIR = os.path.join(BASE_DIR, '../resources/sample_abstracts/')
OUTPUT_DIR = os.path.join(BASE_DIR, '../outputs/')
GRAPH_DIR = os.path.join(BASE_DIR, '../outputs/graphs/')

GRAPH_NAME = 'graph-test2.json'

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
        print(response_text)
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

def process_abstract(abstract_file):
    """
    Process a single abstract file by making three API calls:
    1. Gather all entities.
    2. Refine entities.
    3. Canonicalize entities.
    
    Args:
        abstract_file (str): The path to the abstract file.
    """
    # Extract file name and base name
    abstract_filename = os.path.basename(abstract_file)
    abstract_name, _ = os.path.splitext(abstract_filename)
    
    # Read the abstract content
    try:
        with open(abstract_file, 'r', encoding='utf-8') as file:
            abstract_content = file.read().strip()
    except Exception as e:
        print(f"Failed to read {abstract_filename}: {e}")
        return

    abstract_content = extract_title_and_abstract(abstract_content)
    
    # Generate timestamp once per file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ============================
    # Step 1: Gather All Entities
    # ============================
    try:
        gather_prompt = load_prompt('gather_all_entities.txt')
        response_text_1 = make_api_call(gather_prompt, additional_inputs={"Abstract": abstract_content})
        save_output(response_text_1, OUTPUT_DIR, f"entities_{abstract_name}_{timestamp}.txt")
        
        json_output_1 = parse_json_response(response_text_1)
        if json_output_1:
            save_output(json_output_1, OUTPUT_DIR, f"entities_{abstract_name}_{timestamp}.json", is_json=True)
        else:
            # Save raw response and skip to next abstract
            save_output(response_text_1, OUTPUT_DIR, f"entities_{abstract_name}_{timestamp}_raw.txt")
            return
    except Exception as e:
        print(f"Error during Step 1 for {abstract_filename}: {e}")
        return
    
    # ============================
    # Step 2: Refine Entities
    # ============================
    try:
        refine_prompt = load_prompt('refine_entities.txt')
        additional_inputs = {
            "Abstract": abstract_content,
            "Entities": json.dumps(json_output_1, indent=4)
        }
        response_text_2 = make_api_call(refine_prompt, additional_inputs=additional_inputs)
        save_output(response_text_2, OUTPUT_DIR, f"refined_entities_{abstract_name}_{timestamp}.txt")
        
        refined_json_output = parse_json_response(response_text_2)
        if refined_json_output:
            save_output(refined_json_output, OUTPUT_DIR, f"refined_entities_{abstract_name}_{timestamp}.json", is_json=True)
        else:
            # Save raw response and skip to next abstract
            save_output(response_text_2, OUTPUT_DIR, f"refined_entities_{abstract_name}_{timestamp}_raw.txt")
            return
    except Exception as e:
        print(f"Error during Step 2 for {abstract_filename}: {e}")
        return
    
    # ============================
    # Step 3: Canonicalize Entities
    # ============================
    try:
        canonicalize_prompt = load_prompt('canonicalize_entities.txt')
        additional_inputs = {
            "Refined Entities": json.dumps(refined_json_output, indent=4)
        }
        response_text_3 = make_api_call(canonicalize_prompt, additional_inputs=additional_inputs)
        save_output(response_text_3, OUTPUT_DIR, f"canonicalized_entities_{abstract_name}_{timestamp}.txt")
        
        canonicalized_json_output = parse_json_response(response_text_3)
        if canonicalized_json_output:
            canonicalized_file_name = f"canonicalized_entities_{abstract_name}_{timestamp}.json"
            save_output(canonicalized_json_output, OUTPUT_DIR, canonicalized_file_name, is_json=True)
        else:
            # Save raw response
            save_output(response_text_3, OUTPUT_DIR, f"canonicalized_entities_{abstract_name}_{timestamp}_raw.txt")
    except Exception as e:
        print(f"Error during Step 3 for {abstract_filename}: {e}")
        return
    
    # ============================
    # Step 4: Gather Relationships
    # ============================
    try:
        gather_relations_prompt = load_prompt('gather_all_relationships.txt')
        additional_inputs = {
            "Abstract" : abstract_content,
            "Canonicalized Entities": json.dumps(canonicalized_json_output, indent=4)
        }
        response_text_4 = make_api_call(gather_relations_prompt, additional_inputs=additional_inputs)
        save_output(response_text_4, OUTPUT_DIR, f"canonicalized_relationships_{abstract_name}_{timestamp}.txt")
        
        relations_json_output = parse_json_response(response_text_4)
        if relations_json_output:
            relations_file_name = f"canonicalized_relationships_{abstract_name}_{timestamp}.json"
            save_output(relations_json_output, OUTPUT_DIR, relations_file_name, is_json=True)
        else:
            # Save raw response
            save_output(response_text_4, OUTPUT_DIR, f"canonicalized_relationships_{abstract_name}_{timestamp}_raw.txt")
    except Exception as e:
        print(f"Error during Step 4 for {abstract_filename}: {e}")
        return
    
    print(f"Processing of {abstract_filename} completed.\n")

    print("Now adding it to graph")
    pubmed_id = abstract_filename.strip('.txt')
    add_extracted_data_to_graph(OUTPUT_DIR, canonicalized_file_name, relations_file_name, pubmed_id)
    print("Graph updated")

def add_extracted_data_to_graph(output_dir, canonicalized_file_name, relations_file_name, pubmed_id):
    nodes_path = os.path.join(output_dir, canonicalized_file_name)
    edges_path = os.path.join(output_dir, relations_file_name)

    graph_path = os.path.join(GRAPH_DIR, GRAPH_NAME)
    graph_utils.add_entities_to_graph(graph_path, nodes_path, pubmed_id)
    graph_utils.add_edges_to_graph(graph_path, edges_path, pubmed_id)

# ============================
# Main Execution
# ============================

def main():
    # Get the list of abstract files
    abstract_files = glob.glob(os.path.join(ABSTRACTS_DIR, '*.txt'))
    
    if not abstract_files:
        print(f"No abstract files found in {ABSTRACTS_DIR}.")
        return
    
    # Create graph file
    graph_utils.create_graph(os.path.join(GRAPH_DIR, GRAPH_NAME))

    
    # Process each abstract file
    for abstract_file in tqdm(abstract_files):
        process_abstract(abstract_file)
    
    print("All processing completed.")

if __name__ == "__main__":
    main()
