This document synthesizes the complete architectural plan for your thesis on **Multilingual Multi-Label Entity Framing**, combining the **Stacked Ensemble (Stacked Generalization)** approach with **Semantic Awareness** derived from the taxonomy. This approach is highly effective for multilingual multi-label classification in news entity framing [cite: 2025-11-30].

---

## 🧠 Complete Architectural Plan: Semantic Stacked Ensemble

The system operates as a **Model Stacking** pipeline, leveraging a "committee of smaller experts" (Tier 1) and training a "final decision-maker" (Tier 2) to synthesize their opinions, which often yields higher performance.

### 1. Tier 1: Diverse Base Learners (The Experts)

The goal of Tier 1 is to maximize predictive diversity by varying the model architecture and, crucially, the final classification head, incorporating semantic knowledge.

#### 1.1 Model Selection and Diversity

| Model | Architecture and Role | Final Layer (Key Difference) | Learning Focus |
| :--- | :--- | :--- | :--- |
| **Model A** | **XLM-RoBERTa-Large:** Robust multilingual baseline. | **Standard Linear Head** (Sigmoid activation). | Statistical correlation; general context patterns. |
| **Model B** | **mDeBERTa-v3-base (The Semantic Expert):** Superior disentangled attention. | **Semantic Similarity Head (Label-Embedding Attention)**. | **Semantic Integrity:** Explicitly compares Entity Context against pre-encoded Taxonomy Descriptions for prediction. |
| **Model C** | **DistilBERT-multilingual:** Lightweight, agile model. | **Standard Linear Head** (Sigmoid activation). | Structural diversity; high-level features. |

#### 1.2 Semantic Integration (Model B Detail)
The "Semantic Similarity Head" in Model B replaces the standard classification layer to connect the input text directly to the taxonomy's meaning.

1.  **Label Embedding Pre-computation:** Descriptions (e.g., "Rebels, revolutionaries, or freedom fighters...") are encoded into static vectors (Semantic Anchors) using a small Sentence-BERT model (e.g., `all-MiniLM-L6-v2`).
2.  **Prediction Mechanism:** The prediction is the result of a **Dot Product (Similarity Search)** between the contextual **Entity Vector** ($\mathbf{V}_{entity}$) and the **frozen Label Matrix** ($\mathbf{W}_{taxonomy}$). This forces the model to learn based on semantic distance, which is highly beneficial for rare class handling and disambiguation (e.g., "Instigator" vs. "Saboteur").

$$\text{Logits} = \mathbf{V}_{entity} \cdot \mathbf{W}_{taxonomy}^T \times \text{Scale}$$

#### 1.3 Tier 1 Training and Output
* **Input:** Text segment with entity markers (e.g., `... <e> war </e> ...`).
* **Output:** Vector of **Logits/Probabilities** over all semantic roles (Coarse + Fine).
* **Crucial Step (Out-of-Fold, OOF):** To train Tier 2 without overfitting, Tier 1 models are trained using **K-Fold Cross-Validation** (e.g., 5 folds). Predictions used for the Tier 2 training set are only generated on the hold-out fold.

---

### 2. Tier 2: The Meta-Learner (The Aggregation Strategies)

Tier 2 synthesizes the outputs of the base models to produce the final, high-confidence prediction.

#### Input to Tier 2 (Meta-Feature Vector)
The outputs of the three Tier 1 models are concatenated to form the input for the Meta-Learner:
$$X_{meta} = [P_{A} \oplus P_{B} \oplus P_{C}]$$
Where $P$ is the vector of probabilities/logits from the respective model.

#### Option A: Classical Neural Aggregator (The Primary Solution)
This is a robust and fast solution, leveraging statistical patterns in the ensemble outputs.

* **Architecture:** A lightweight **Multi-Layer Perceptron (MLP)**.
* **Process:** The MLP uses a hidden layer (ReLU + BatchNorm) to learn non-linear relationships (e.g., "trust Model B when Model A and C disagree") before a final **Sigmoid output layer**.

$$P_{final} = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot X_{meta} + b_1) + b_2)$$

#### Option B: LLM "Semantic Judge" (Experimental Alternative)
This is an advanced, experimental option that leverages the reasoning power of a larger, free LLM (e.g., **OpenLLaMA 3B/7B or Mistral-7B**) to resolve hard cases.

* **Input:** Structured **zero-shot prompt**:
    * The original sentence/paragraph.
    * The top predictions and confidence scores from Models A, B, and C.
    * The **Taxonomy Definitions** of the contending roles.
* **Process:** The LLM acts as a reasoned judge, weighing the expert opinions against the semantic rules to generate the final classification. This leverages "common sense" reasoning that smaller models lack.

---

## 3. Full Workflow and Recommendation

### Step-by-Step Summary 
1.  **Data Split:** Divide the training data into K folds (e.g., 5 folds).
2.  **Train Tier 1:** Train Models A, B, and C on 4 folds, generating OOF predictions on the 5th until the entire training set has Meta-Features. Train A, B, and C on the full training set for final inference.
3.  **Train Tier 2 (Option A):** Train the MLP using the OOF predictions ($X_{meta}$) as input and the Gold Labels as the target.
4.  **Setup Tier 2 (Option B):** Either use the LLM in **Zero-shot** mode or use **LoRA** to fine-tune it specifically on the prompt format.
5.  **Final Inference:** New Text $\rightarrow$ Models A, B, C $\rightarrow$ Concatenated Predictions $\rightarrow$ Tier 2 (MLP or LLM) $\rightarrow$ **Final Role Assignment**.

### Recommendation
It is highly recommended to **implement both Option A and Option B**. Option A serves as the **scientifically rigorous primary solution**, while Option B provides an **experimental comparison** to assess if explicit, LLM-based reasoning improves performance on challenging disagreements.

---
---
---
---
---
This project structure outlines the development process for your **Semantic Stacked Ensemble** model for Multilingual Entity Framing. The phases are designed sequentially, emphasizing data integrity, architectural diversity, and rigorous evaluation.  

## I. Data Preparation and Taxonomy Integration

This initial phase focuses on preparing the raw data into standardized formats and encoding the semantic knowledge from your taxonomy.

### 1. Data Loading and Standardization
* **Source Data Ingestion:** Load the raw multilingual corpus (Bulgarian, English, Hindi, Portuguese, Russian) and the corresponding document-level annotations.
* **Annotation Expansion:** Implement the data transformation logic (as discussed in your code sample) to expand document-level annotations into **sentence-level units**. This is critical for training the smaller, context-focused Tier 1 models.
* **Label Preparation:** Load the `taxonomy.json` file.
    * Map the fine-grained semantic roles to unique numerical IDs.
    * Convert the categorical labels for each entity into a **Multi-Label (Binary) Vector** format, where each element corresponds to a specific role.

### 2. Dataset Partitioning
* **Full Train/Test Split:** Create the definitive Test Set (using the provided development data, if applicable) for final, unbiased evaluation.
* **K-Fold Definition:** Define a **K-Fold Cross-Validation** strategy (e.g., K=5) for the remaining Training Data. This is essential for generating the unbiased **Out-of-Fold (OOF) predictions** needed for training the Tier 2 Meta-Learner.

### 3. Taxonomy Semantic Encoding
* **Description Extraction:** Extract the descriptive text for every semantic role from `taxonomy.json`.
* **Anchor Vector Generation:** Use an external Sentence Transformer (e.g., `all-MiniLM-L6-v2`) to encode these descriptions into **static, high-dimensional label embedding vectors** ($\mathbf{W}_{taxonomy}$). These vectors will serve as the **Semantic Anchors** for Model B.

***

## II. Tier 1 Development: Base Learners

This phase involves training the three diverse base models and collecting their predictions on the entire dataset. 

### 1. Model Configuration and Training
For each model (A, B, C):
* **Tokenization:** Load the multilingual tokenizer (e.g., XLM-R) and add custom entity tokens (`[ENTITY]`, `[/ENTITY]`).
* **Architecture Setup:** Instantiate the respective base model (XLM-R, mDeBERTa, DistilBERT).
    * **Model A/C:** Attach a **Standard Linear Classification Head** (Sigmoid activation).
    * **Model B (The Semantic Expert):** Replace the standard head with the **Semantic Similarity Head** ($\mathbf{W}_{taxonomy}$), initialized with the pre-computed label embeddings.
* **OOF Training Loop:** Run the **K-Fold Cross-Validation** process:
    1.  Train the model on **K-1** folds.
    2.  Use the trained model to predict on the held-out **Kth** fold.
    3.  Collect and store the raw **Logits/Probabilities** for the Kth fold.
* **Final Training:** After the OOF process is complete, train a final version of Models A, B, and C on the **entire training set** for use during final inference.

### 2. Prediction Collection
* **OOF Prediction Collection (Training Tier 2):** Concatenate all OOF predictions from the K-folds for Models A, B, and C to create the **Training Meta-Feature Set** ($X_{meta}$).
* **Test Prediction Collection (Inference):** Use the final, fully-trained Models A, B, and C to generate predictions (Logits/Probabilities) on the reserved **Test Set**.

***

## III. Tier 2 Development: Meta-Learner

This phase uses the collected OOF predictions to train the aggregator models.

### 1. Training the Classical Neural Aggregator (Option A)
* **Input Preparation:** The training data is the **OOF Meta-Feature Set** ($X_{meta}$). The target is the original Gold Label Multi-Label Vectors.
* **Model Architecture:** Define a simple **Multi-Layer Perceptron (MLP)** with one or two hidden layers (using ReLU and BatchNorm).
* **Training:** Train the MLP using **Binary Cross-Entropy Loss** and the Adam optimizer.
* **Final Output:** A trained MLP capable of synthesizing the statistical and semantic evidence from Tier 1.

### 2. Setting Up the LLM Semantic Judge (Option B)
* **Model Setup:** Initialize the chosen LLM (e.g., OpenLLaMA 3B/7B or Mistral-7B).
* **Prompt Engineering:** Design the detailed zero-shot prompt template, which includes the raw text, the entity, the conflicting predictions from Tier 1 models, and the relevant Taxonomy Definitions.
* **Optional Fine-Tuning:** For better performance, consider using **LoRA (Low-Rank Adaptation)** to fine-tune the LLM specifically on the prompt/output format, teaching it to reliably return the desired structure (e.g., a JSON list of predicted roles).

***

## IV. Final Evaluation and Analysis

The final phase uses the trained ensemble components to generate the submission and draw conclusions.

### 1. Test Set Inference
* **Tier 1 Execution:** Run the final versions of Models A, B, and C on the **Test Set** to generate the **Test Meta-Feature Set**.
* **Tier 2 Execution:**
    * **Option A:** Pass the Test Meta-Feature Set through the **Trained MLP**.
    * **Option B (Comparative):** Pass the raw text and Tier 1 predictions through the **LLM Semantic Judge** via the structured prompt.
* **Final Output:** Generate the predicted Multi-Label Vector for every entity in the Test Set for both Option A and Option B.

### 2. Metric Calculation
* **Primary Metrics:** Calculate standard multi-label metrics (e.g., **micro-F1, Precision, Recall, Accuracy**) on the Test Set for both Option A and Option B against the true multi-label vectors.
* **Granularity:** Evaluate performance at both the **Coarse-Grained** (Protagonist, Antagonist, etc.) and the **Fine-Grained** (Guardian, Tyrant, etc.) levels.
* **Comparative Analysis:** Calculate performance metrics broken down by:
    * Individual Tier 1 Models (Baseline).
    * Ensemble Option A (MLP).
    * Ensemble Option B (LLM Judge).
    * **Per-Language** performance to assess cross-lingual transfer efficiency.

### 3. Conclusion and Thesis Writing
* **Discussion:** Analyze whether the **Semantic Expert (Model B)** contributed unique value to the ensemble compared to the standard models.
* **Comparison:** Conclude whether the **LLM Semantic Judge (Option B)**'s reasoning capabilities successfully outperformed the simpler statistical combination of the **MLP (Option A)**, especially on hard-to-classify or rare examples.