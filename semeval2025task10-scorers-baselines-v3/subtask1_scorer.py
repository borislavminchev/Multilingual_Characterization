import logging
import argparse
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import re

logger = logging.getLogger("entity_framing_scorer")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.WARNING)  

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


MAIN_ROLES = ['Protagonist', 'Antagonist', 'Innocent']
FINE_GRAINED_ROLES = ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous',
                           'Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 'Terrorist', 'Deceiver', 'Bigot',
                           'Forgotten', 'Exploited', 'Victim', 'Scapegoat']


def read_file(file,file_dict={}):
    file_dict = {}
    with open(file, encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) < 5:
                error_message = f"Error: Line {line_num + 1} in {file} is not a valid TSV format or has too few columns."
                logger.error(error_message)
                raise ValueError(error_message)

            if line_num == 0:
                if parts == ["article_id", "entity_mention", "start_offset", "end_offset", "main_role", "fine_grained_roles"]:
                    
                    continue

           
            article_id = parts[0]
            entity_mention = re.sub("\"","",parts[1])
            start_offset = parts[2]
            end_offset = parts[3]
            main_role = parts[4]
            fine_grained_roles = parts[5:]  
            key = (article_id, entity_mention, start_offset, end_offset)

            if key in file_dict:
                error_message = f"Error: Duplicate key found on line {line_num + 1} in {file}. Key: {key}"
                logger.error(error_message)
                raise ValueError(error_message)

            file_dict[key] = (main_role, fine_grained_roles)
    
    
    return file_dict


def check_file_format(gold_dict, pred_dict):
    """Check if the prediction file is in the expected format."""
    errors = []

    


    if len(gold_dict) != len(pred_dict):
        error_message = f"File size mismatch: Gold file has {len(gold_dict)} entries, but prediction file has {len(pred_dict)} entries."
        logger.error(error_message)
        errors.append(error_message)


    if gold_dict.keys() != pred_dict.keys():
        missing_in_pred = set(gold_dict.keys()) - set(pred_dict.keys())
        extra_in_pred = set(pred_dict.keys()) - set(gold_dict.keys())

        if missing_in_pred:
            error_message = f"Missing entries in prediction file: {missing_in_pred}"
            logger.error(error_message)
            errors.append(error_message)
        if extra_in_pred:
            error_message = f"Extra entries in prediction file: {extra_in_pred}"
            logger.error(error_message)
            errors.append(error_message)

    else:
        for key in gold_dict.keys():
            pred_main_role, pred_fine_grained_roles = pred_dict[key]
            if pred_main_role not in MAIN_ROLES:
                error_message = f"Invalid main role '{pred_main_role}' in prediction for key: {key}"
                logger.error(error_message)
                errors.append(error_message)
            
            for role in pred_fine_grained_roles:
                if role not in FINE_GRAINED_ROLES:
                    error_message = f"Invalid fine-grained role '{role}' in prediction for key: {key}"
                    logger.error(error_message)
                    errors.append(error_message)

    return errors

def exact_match_ratio(gold_labels, pred_labels):
    """Calculate Exact Match Ratio based only on secondary labels."""
    exact_matches = 0
    for key in gold_labels.keys():
        if set(gold_labels[key][1]) == set(pred_labels[key][1]):
            exact_matches += 1
    total = len(gold_labels)
    ratio = exact_matches / total if total > 0 else 0
    
    return ratio


def evaluate_fine_grained_metrics(gold_dict, pred_dict):
    """Evaluate micro and macro F1, precision, and recall for fine-grained roles."""
    
    mlb = MultiLabelBinarizer()
    
    all_labels = set(role for (_, roles) in gold_dict.values() for role in roles) | \
                 set(role for (_, roles) in pred_dict.values() for role in roles)
    
    mlb.fit([list(all_labels)])  # Fit the MultiLabelBinarizer
    
    gold_values = [mlb.transform([roles])[0] for _, roles in gold_dict.values()]
    pred_values = [mlb.transform([roles])[0] for _, roles in pred_dict.values()]

    micro_f1 = f1_score(gold_values, pred_values, average="micro", zero_division=1)
    macro_f1 = f1_score(gold_values, pred_values, average="macro", zero_division=1)
    micro_precision = precision_score(gold_values, pred_values, average="micro", zero_division=1)
    macro_precision = precision_score(gold_values, pred_values, average="macro", zero_division=1)
    micro_recall = recall_score(gold_values, pred_values, average="micro", zero_division=1)
    macro_recall = recall_score(gold_values, pred_values, average="macro", zero_division=1)

    
    return micro_f1, macro_f1, micro_precision, macro_precision, micro_recall, macro_recall


def evaluate_main_role_accuracy(gold_dict, pred_dict):
    """Evaluate accuracy for the main role."""
    
    gold_main_roles = [main_role for main_role, _ in gold_dict.values()]
    pred_main_roles = [main_role for main_role, _ in pred_dict.values()]
    
    accuracy = accuracy_score(gold_main_roles, pred_main_roles)
    
    return accuracy


def main(gold_file_path, pred_file_path):
    """Main function to execute the scorer."""
    
    gold_dict = read_file(gold_file_path)
    pred_dict = read_file(pred_file_path)

    format_errors = check_file_format(gold_dict, pred_dict)
    if format_errors:
        for error in format_errors:
            logger.error(error)
        sys.exit("Prediction file format check failed.")

    emr = exact_match_ratio(gold_dict, pred_dict)
    
    micro_f1, macro_f1, micro_precision, macro_precision, micro_recall, macro_recall = evaluate_fine_grained_metrics(gold_dict, pred_dict)
    

    
    main_role_accuracy = evaluate_main_role_accuracy(gold_dict, pred_dict)

    print(f"{emr:.4f}\t{micro_precision:.4f}\t{micro_recall:.4f}\t{micro_f1:.4f}\t{main_role_accuracy:.4f}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Entity Framing Scorer: Evaluate predictions for entity role classification."
    )
    parser.add_argument("--gold_file_path", '-g', type=str, required=True, help="Path to the file with gold annotations.")
    parser.add_argument("--pred_file_path", '-p', type=str, required=True, help="Path to the file with predictions")
    parser.add_argument("--log_to_file", "-l", action='store_true', default=False, help="Set flag if you want to log the execution to a file.")
    
    args = parser.parse_args()
    
    if args.log_to_file:
        output_log_file = args.pred_file_path + ".log"
        
        fileLogger = logging.FileHandler(output_log_file)
        fileLogger.setLevel(logging.DEBUG)
        fileLogger.setFormatter(formatter)
        logger.addHandler(fileLogger)
        logger.setLevel(logging.DEBUG)

    main(args.gold_file_path, args.pred_file_path)


# Example usage:
# python subtask1_scorer.py -g path/to/gold.tsv -p path/to/pred.tsv
# To log the output to a file in addition to the console:
# python subtask1_scorer.py -g path/to/gold.tsv -p path/to/pred.tsv -l