import json
import os

def process_file(input_filename, output_filename):
    data = []

    with open(input_filename, 'r', encoding='utf-8') as f:
        block_lines = []
        for line in f:
            line = line.rstrip('\n')
            if line.strip() == '':
                if block_lines:
                    # Process the current block
                    record = process_block(block_lines)
                    if record:
                        data.append(record)
                    block_lines = []
            else:
                block_lines.append(line)
        # Process the last block if the file doesn't end with an empty line
        if block_lines:
            record = process_block(block_lines)
            if record:
                data.append(record)

    # Write the processed data to the JSON output file
    with open(output_filename, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=2, ensure_ascii=False)

def process_block(block_lines):
    pbid = None
    title = ''
    abstract_text = ''
    entities = {}          # Maps entity IDs to their names and types
    entities_set = set()   # To ensure uniqueness of nodes
    nodes = []
    edges = []

    for line in block_lines:
        if '|t|' in line:
            parts = line.split('|t|', 1)
            if len(parts) == 2:
                pbid, title = parts
        elif '|a|' in line:
            parts = line.split('|a|', 1)
            if len(parts) == 2:
                pbid_a, abstract_text = parts
                if pbid_a != pbid:
                    print(f'Warning: pbid mismatch in abstract line: {pbid_a} != {pbid}')
        else:
            tokens = line.strip().split('\t')
            if len(tokens) < 2:
                print(f'Warning: Unrecognized line (too few tokens): {line}')
                continue

            # Determine if the line is an entity or a relationship
            if tokens[1].isdigit() and (len(tokens) >= 3 and tokens[2].isdigit()):
                # Entity line
                if len(tokens) >= 6:
                    # Format: id \t num \t num \t name \t type \t entity_id
                    entity_id = tokens[5]
                    entity_type = tokens[4]
                    name = tokens[3]
                elif len(tokens) == 5:
                    # Format: id \t num \t num \t name \t type
                    entity_id = None
                    entity_type = tokens[4]
                    name = tokens[3]
                else:
                    print(f'Warning: Unexpected number of tokens in entity line: {line}')
                    continue

                # Add to entities dictionary if entity_id is present
                if entity_id:
                    if entity_id not in entities:
                        entities[entity_id] = {'name': name, 'type': entity_type}
                # Add to nodes set to ensure uniqueness
                node_key = (name, entity_type)
                if node_key not in entities_set:
                    entities_set.add(node_key)
                    nodes.append({'name': name, 'type': entity_type})
            else:
                # Relationship line
                if len(tokens) >= 5:
                    # Format: id \t relationship_type \t id1 \t id2 \t novel_no
                    relationship_type = tokens[1]
                    id1 = tokens[2]
                    id2 = tokens[3]
                    # novel_no = tokens[4]  # Currently not used

                    # Retrieve entity names based on IDs
                    name1 = entities.get(id1, {}).get('name', id1)
                    name2 = entities.get(id2, {}).get('name', id2)

                    edges.append({
                        'source': name1,
                        'target': name2,
                        'type': relationship_type
                    })
                else:
                    print(f'Warning: Unrecognized relationship line: {line}')

    if not pbid:
        print('Warning: Missing pbid in block. Skipping block.')
        return None

    record = {
        'pbid': pbid,
        'title': title,
        'abstract_text': abstract_text,
        'entities': nodes,
        'relationships': edges
    }

    return record

if __name__ == "__main__":
    # Replace 'input_file.txt' with your actual input filename
    # Replace 'output_file.json' with your desired output filename
    # get input file path
    file_name = 'test1'
    input_file_path = os.path.join(os.path.dirname(__file__), f'..\\..\\resources\\biored_data\\{file_name}.txt')
    output_file_path = os.path.join(os.path.dirname(__file__), f'..\\..\\resources\\biored_data\\{file_name}_formatted.json')
    process_file(input_file_path, output_file_path)
