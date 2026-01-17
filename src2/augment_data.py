
import pandas as pd
from google.generativeai.client import configure 
from google.generativeai.generative_models import GenerativeModel
import time
import os
from tqdm import tqdm
import re
import csv

from data_utils import load_data

# -----------------------------
# CONFIGURATION
# -----------------------------
# Get your free key from: https://aistudio.google.com/
GOOGLE_API_KEY = "AIzaSyBhjshb40naDxnWcb339IwtAIfX2sxtCGE" 

configure(api_key=GOOGLE_API_KEY)

# We use Gemma 3 27B as it has the high 14.4k daily limit
# Note: If 'gemma-3-27b-it' is available, use that (Instruct version is better). 
# Otherwise 'gemma-3-27b' is fine.
MODEL_NAME = 'gemma-3-27b-it' 

def is_invalid_rationale(rationale, mention):
    if rationale is None:
        return True

    rationale = rationale.strip()

    return (
        rationale == mention or
        rationale == "" or
        len(rationale) <= len(mention)
    )

# -----------------------------
# NEW: REGEX PARSER (No JSON)
# -----------------------------
def parse_delimited_output(text, batch_ids):
    """
    Parses output that looks like:
    ID: 0
    RATIONALE: The army attacked the border...
    
    Returns a dictionary mapping {id: rationale}
    """
    results = {}
    
    # We iterate through the IDs we expect to find
    for bid in batch_ids:
        # Regex explanation:
        # 1. Search for "ID: <bid>" (flexible whitespace)
        # 2. Look for "RATIONALE:" tag
        # 3. Capture everything until the next "ID:" or end of string
        #
        pattern = fr"ID:\s*{bid}\s*[\n\r]*RATIONALE:\s*(.*?)(?=ID:\s*\d|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            clean_text = match.group(1).strip()
            # Clean up common artifacts if model is chatty
            clean_text = clean_text.replace("is valid: true", "").strip()
            results[bid] = clean_text
        else:
            results[bid] = None # Mark as missing to handle later

    return results

# -----------------------------
# BATCH PROMPT (TEXT ONLY)
# -----------------------------
def process_batch(batch_df, model):
    batch_text = ""
    # We use a simple text format
    for idx, row in batch_df.iterrows():
        batch_text += f"""
        ID: {idx}
        ENTITY: {row['mention']}
        ROLE: {row['labels']}
        TEXT: {row['text']}
        
        """

    # Instruction is much simpler now. No JSON schema.
    prompt = f"""
    You are an Extractor.
    For each item below, extract the exact text (rationale) from the TEXT that proves the ENTITY has the ROLE.
    
    RULES:
    1. Output format MUST be:
       ID: <number>
       RATIONALE: <extracted text>
    2. NO BARE NAMES: 
       - Bad Rationale: "NATO"
       - Good Rationale: "NATO sent troops to the border"
    3. Do not use Markdown or JSON. Just plain text.
    4. The rationale MUST include the verb/action/description that justifies the role of the entity.
    
    ITEMS:
    {batch_text}
    """

    try:
        response = model.generate_content(prompt)
        # Pass the list of expected IDs to the parser
        return parse_delimited_output(response.text, batch_df.index.tolist())
    except Exception as e:
        error_str = str(e)
        if "429" in error_str:
            match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
        if match:
            wait_time = float(match.group(1))
            print(f"⚠️ Rate limit hit. Waiting for {wait_time}s...")
            time.sleep(wait_time)
        print(f"⚠️ Batch Generation Error")
        return None

# -----------------------------
# MAIN LOOP
# -----------------------------
def expand_dataset_gemma(df, output_path, batch_size=15):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model = GenerativeModel(MODEL_NAME)

    if os.path.exists(output_path):
        processed_df = pd.read_csv(output_path)
        # Filter out rows that are already in the output file
        existing_ids = set(processed_df.index) 
        # Note: Ideally your CSV has an ID column. If not, we assume order.
        # For safety, let's just slice:
        start_index = len(processed_df)
        results = processed_df.to_dict("records")
        print(f"🔄 Resuming from row {start_index}")
    else:
        results = []
        start_index = 0
    
    df_remaining = df.iloc[start_index:]

    # 15k TPM Strategy: Small Batch (2) + Steady Wait
    for i in tqdm(range(0, len(df_remaining), batch_size), desc="Processing Batches"):
        batch = df_remaining.iloc[i : i + batch_size]
        
        success = False
        for attempt in range(12): 
            batch_output = process_batch(batch, model)
            
            if batch_output:
                # Process results
                for idx, row in batch.iterrows():
                    rationale = batch_output.get(idx)
                    
                    # Validation Logic
                    is_valid = True
                    if not rationale or rationale == "INVALID" or len(rationale) <= len(row['mention']) + 2:
                        is_valid = False
                        if not rationale: rationale = row['mention'] # Fallback

                    # Offset calculation
                    start_char = row["text"].find(rationale)
                    
                    new_row = row.to_dict()
                    new_row.update({
                        "rationale_text": rationale,
                        "rationale_start": start_char,
                        "rationale_end": start_char + len(rationale) if start_char != -1 else -1,
                        "valid_extraction": is_valid
                    })
                    results.append(new_row)
                
                success = True
                break
            else:
                # Wait logic for 429 errors
                wait_time = (5 ** (attempt % 3) + 1) 
                print(f"⚠️ Error. Cooling down for {wait_time}s...")
                time.sleep(wait_time)
        
        if not success:
            print(f"❌ Failed batch {i}.")

        # 15k TPM Pacing:
        # Batch size 2 is small. 3 seconds sleep is safe.
        time.sleep(3.0) 

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
    print(f"✅ Completed. Total rows: {len(final_df)}")

    invalid_rate = sum(
        is_invalid_rationale(r["rationale_text"], r["mention"])
        for r in results
    ) / len(results)
    print(f"🚨 Invalid rationale rate: {invalid_rate:.2%}") 

    return final_df

# Usage
train_df, val_df, test_df = load_data(mode="paragraph")
expand_dataset_gemma(train_df, "./paragraph/augmented_train.csv")
expand_dataset_gemma(val_df, "./paragraph/augmented_val.csv")
expand_dataset_gemma(test_df, "./paragraph/augmented_test.csv")