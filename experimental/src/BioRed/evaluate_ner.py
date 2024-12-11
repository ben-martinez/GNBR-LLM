import json
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import os

def compute_metrics(file1, file2):
    # Load JSON files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Ensure both files are aligned by `pbid` and `abstract_text`
    pbid_to_entities_1 = {entry['pbid']: set((e['name'], e['type']) for e in entry['entities']) for entry in data1}
    pbid_to_entities_2 = {entry['pbid']: set((e['name'], e['type']) for e in entry['entities']) for entry in data2}

    all_pbids = set(pbid_to_entities_1.keys()).intersection(set(pbid_to_entities_2.keys()))
    
    # Initialize metrics
    total_tp, total_fp, total_fn = 0, 0, 0

    for pbid in all_pbids:
        entities_1 = pbid_to_entities_1.get(pbid, set())
        entities_2 = pbid_to_entities_2.get(pbid, set())

        tp = len(entities_1.intersection(entities_2))  # True Positives
        fp = len(entities_1.difference(entities_2))    # False Positives
        fn = len(entities_2.difference(entities_1))    # False Negatives

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Compute precision, recall, and F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def compute_rtpr(file1, file2):
    # Load JSON files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Parse entities by pbid
    pbid_to_entities_1 = {entry['pbid']: set((e['name'], e['type']) for e in entry['entities']) for entry in data1}
    pbid_to_entities_2 = {entry['pbid']: set((e['name'], e['type']) for e in entry['entities']) for entry in data2}

    all_pbids = set(pbid_to_entities_1.keys()).intersection(set(pbid_to_entities_2.keys()))
    
    total_tp_file1 = 0
    total_tp_file2 = 0

    for pbid in all_pbids:
        entities_1 = pbid_to_entities_1.get(pbid, set())
        entities_2 = pbid_to_entities_2.get(pbid, set())

        # True positives for each file
        tp_file1 = len(entities_1.intersection(entities_2))  # TP from File 1
        tp_file2 = len(entities_2)  # TP from File 2 (assumes File 2 is reference)

        total_tp_file1 += tp_file1
        total_tp_file2 += tp_file2

    rtpr = total_tp_file1 / total_tp_file2 if total_tp_file2 > 0 else 0
    return rtpr

if __name__ == "__main__":
    model_output_filename = 'biored_tagged_ents_20241203183625_full_test.json'
    gold_filename = 'full_test_formatted.json'

    model_output_filepath = os.path.join(os.path.dirname(__file__), f'..\\..\\outputs\\biored_outputs\\ents\\{model_output_filename}')
    gold_output_filepath = os.path.join(os.path.dirname(__file__), f'..\\..\\resources\\biored_data\\{gold_filename}')

    precision, recall, f1 = compute_metrics(model_output_filepath, gold_output_filepath)


    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")