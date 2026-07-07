# ./defensive_baseline.py

import os
import re
import json
import torch
import logging
import warnings
import argparse
import traceback
import numpy as np
from typing import Dict, Tuple, List
from tqdm import tqdm
from logger_config import logger

warnings.filterwarnings('ignore')

WEIGHTS_DIR = "./baseline_layer_weights"
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "baseline_weights.json")

class DefenseLayer:
    """
    Fuzzy, Weighted Multi-Layer BASELINE-Defense Pipeline.
    Uses a trained linear combination of layer triggers to evaluate prompts softly.
    """
    def __init__(self, config=None, training_mode=False):
        logger.info("="*60)
        if not training_mode:
            logger.info("[BASELINE-DEFENSE] 🛡️  Initializing Fuzzy Weighted Baseline Defense for INFERENCE...")
        else:
            logger.info("[BASELINE-DEFENSE] 🛠️  Initializing Baseline Defense for TRAINING MODE...")
        
        self.config = config or {}
        self.training_mode = training_mode
        
        # ==========================================
        # LAYER 1: STATIC KEYWORDS & OBFUSCATION 
        # ==========================================
        self.THREAT_PATTERNS = [
            r'ignore\s+(previous|all|above)\s+instructions?', r'forget\s+(previous|all|everything)',
            r'new\s+instructions?:?', r'you\s+are\s+now', r'system\s+prompt',
            r'reveal\s+(your|the)\s+prompt', r'show\s+(your|me\s+the)\s+prompt',
            r'(DAN|evil|jailbreak)\s+mode', r'pretend\s+(you|to)\s+(are|be)',
            r'act\s+as\s+(if|a)', r'sudo\s+', r'developer\s+mode', r'bypass\s+',
            r'print\s+your\s+(instructions|rules)'
        ]
        self.OBFUSCATION_PATTERNS = [
            r'[il1][gq]n[o0]r[e3]', r'[il1]n[s\$]tr[u_]ct[il1][o0]n', 
            r'pr[o0]mpt', r's[vy][s\$]t[e3]m'
        ]
        self.compiled_threats = [re.compile(p, re.IGNORECASE) for p in self.THREAT_PATTERNS]
        self.compiled_obfuscation = [re.compile(p, re.IGNORECASE) for p in self.OBFUSCATION_PATTERNS]
        logger.info(f"[BASELINE-DEFENSE] 🔍 Compiled {len(self.compiled_threats)} Threat Patterns and {len(self.compiled_obfuscation)} Obfuscation Patterns.")

        # ==========================================
        # LAYER 2: SEMANTIC TOXICITY (CPU-Bound)
        # ==========================================
        try:
            logger.info("[BASELINE-DEFENSE] ⚙️  Loading Toxicity Classifier...")
            from transformers import pipeline
            
            # --- GPU ACCELERATION FIX ---
            # Use device=0 for the first GPU if available
            gpu_device = 0 if torch.cuda.is_available() else -1
            
            self.toxicity_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert", 
                device=gpu_device  # Changed from -1 to 0
            )
            
            if gpu_device == 0:
                logger.info("[BASELINE-DEFENSE] ⚡ Toxicity Classifier loaded on GPU.")
            else:
                logger.info("[BASELINE-DEFENSE] ✅ Toxicity Classifier loaded on CPU.")
                
        except Exception as e:
            logger.warning(f"[BASELINE-DEFENSE] ⚠️ Failed to load Toxicity Classifier: {e}")
            self.toxicity_classifier = None

        # ==========================================
        # LAYER 3/4: OUTPUT GUARDS (Payload & Leaks)
        # ==========================================
        self.THREAT_DATABASE = {
            "CRITICAL_HARM": ["bomb", "ied", "shrapnel", "anthrax", "sarin", "suicide", "murder", "bioweapon"],
            "CYBER_ATTACK": ["sql injection", "buffer overflow", "rce", "xss", "keylogger", "rootkit", "ransomware", "zero-day"],
            "ECONOMIC_HARM": ["insider trading", "ponzi", "pyramid scheme", "embezzle", "counterfeit", "tax evasion"],
            "DISINFORMATION": ["illuminati", "flat earth", "hoax", "fake news", "deepfake", "vaccine autism"]
        }
        self.compiled_payloads = [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for words in self.THREAT_DATABASE.values() for w in words]

        self.LEAK_PATTERNS = [
            r'<\|system\|>', r'<\|im_start\|>system', r'system\s*[:=]\s*["\']',
            r'my\s+instructions\s+(are|were|say)', r'I\s+was\s+(told|instructed|programmed)\s+to'
        ]
        self.compiled_leaks = [re.compile(p, re.IGNORECASE) for p in self.LEAK_PATTERNS]
        logger.info(f"[BASELINE-DEFENSE] 🛡️  Compiled {len(self.compiled_payloads)} Payload Guards and {len(self.compiled_leaks)} System Leak Guards.")

        # ==========================================
        # LOAD TRAINED WEIGHTS
        # ==========================================
        self.weights = {"input_weights": [0.5, 0.5, 0.5], "input_bias": -1.0, "threshold": 0.5}
        if not training_mode:
            self._load_weights()
            
        logger.info("="*60)

    def _load_weights(self):
        """Loads the pre-trained weights for the fuzzy logic."""
        logger.info(f"[BASELINE-DEFENSE] 📂 Attempting to load weights from: {WEIGHTS_FILE}")
        if os.path.exists(WEIGHTS_FILE):
            try:
                with open(WEIGHTS_FILE, 'r') as f:
                    data = json.load(f)
                    self.weights = data
                logger.info(f"[BASELINE-DEFENSE] ✅ Successfully loaded weights:")
                logger.info(f"   -> Weights (Threat, Obf, Tox): {self.weights['input_weights']}")
                logger.info(f"   -> Bias: {self.weights['input_bias']:.4f} | Threshold: {self.weights['threshold']}")
            except Exception as e:
                logger.error(f"[BASELINE-DEFENSE] ❌ Failed to parse weights: {e}. Using default weak fallbacks.")
        else:
            logger.warning("[BASELINE-DEFENSE] ⚠️ No trained weights found! Using default weak fallbacks. Run with --baseline-weights-train first.")

    def _normalize_text(self, text: str) -> str:
        normalized = text.lower()
        substitutions = {'0':'o', '1':'i', '3':'e', '4':'a', '5':'s', '7':'t', '8':'b', '@':'a', '$':'s', '!':'i', '|':'i'}
        for char, replacement in substitutions.items():
            normalized = normalized.replace(char, replacement)
        return re.sub(r'[^a-z0-9\s]', '', re.sub(r'\s+', ' ', normalized))

    def _extract_input_features(self, text: str) -> List[float]:
        """Converts raw text into a 3D feature vector [Threat_Match, Obfuscation_Match, Toxicity_Score]"""
        logger.debug("[BASELINE-DEFENSE] 🔎 Extracting features from input...")
        
        f_threat = 1.0 if any(p.search(text) for p in self.compiled_threats) else 0.0
        
        normalized = self._normalize_text(text)
        f_obf = 1.0 if any(p.search(normalized) for p in self.compiled_obfuscation) else 0.0
        
        f_tox = 0.0
        if self.toxicity_classifier:
            try:
                res = self.toxicity_classifier(text[-2000:][:512])[0]
                if res['label'].upper() in ['TOXIC', 'TOXICITY']:
                    f_tox = float(res['score'])
            except Exception as e:
                logger.debug(f"[BASELINE-DEFENSE] Toxicity classification failed: {e}")
                
        logger.info(f"[BASELINE-DEFENSE] 📊 Extracted Features -> Threat: {f_threat}, Obfuscation: {f_obf}, Toxicity: {f_tox:.4f}")
        return [f_threat, f_obf, f_tox]

    def _extract_output_features(self, text: str) -> List[float]:
        """Converts output response into a 2D feature vector [Payload_Match, System_Leak]"""
        f_leak = 1.0 if any(p.search(text) for p in self.compiled_leaks) else 0.0
        f_payload = 1.0 if any(p.search(text) for p in self.compiled_payloads) else 0.0
        logger.info(f"[BASELINE-DEFENSE] 📊 Extracted Output Features -> System Leak: {f_leak}, Malicious Payload: {f_payload}")
        return [f_leak, f_payload]

    # ==========================================
    # PIPELINE INTEGRATION METHODS
    # ==========================================
    def process_input(self, context: str) -> Tuple[bool, str]:
        """Orchestrator Entrypoint using Fuzzy Weighted Scoring."""
        if not isinstance(context, str): context = str(context) if context else ""
        if context.lower() == "nan" or not context: 
            logger.debug("[BASELINE-DEFENSE] Empty or NaN context received. Passing.")
            return False, ""

        # Extract latest turn
        parts = context.rsplit("User:", 1)
        latest_turn = parts[-1].strip() if len(parts) > 1 else context.strip()
        
        logger.info(f"[BASELINE-DEFENSE] 📥 Processing Input (Length: {len(latest_turn)} chars)...")

        # Extract features and apply Logistic Regression weights
        features = self._extract_input_features(context)
        
        w = np.array(self.weights["input_weights"])
        b = self.weights["input_bias"]
        
        # Calculate logit and apply sigmoid for a 0-1 probability score
        logit = np.dot(w, features) + b
        prob = 1 / (1 + np.exp(-logit))
        
        logger.info(f"[BASELINE-DEFENSE] 🧮 Math -> Logit: {logit:.4f} | Final Prob: {prob:.4f} | Threshold: {self.weights['threshold']}")
        
        if prob > self.weights["threshold"]:
            logger.warning(f"[BASELINE-DEFENSE] 🛑 BLOCKED | Fuzzy Score ({prob:.2f}) exceeded threshold.")
            return True, latest_turn

        logger.info("[BASELINE-DEFENSE] ✅ PASSED | Prompt marked as safe by Baseline.")
        return False, latest_turn

    def process_output(self, response: str) -> Tuple[bool, str]:
        """Orchestrator Exitpoint."""
        if not isinstance(response, str): return False, response
        
        logger.info(f"[BASELINE-DEFENSE] 📤 Processing Output (Length: {len(response)} chars)...")

        # For outputs, if a raw payload or leak is detected, we hard-block 
        features = self._extract_output_features(response)
        if sum(features) > 0:
            logger.warning("[BASELINE-DEFENSE] 🛑 OUTPUT BLOCKED | Hard-Block triggered by Leak/Payload detection.")
            return True, "I cannot fulfill this request."

        # Fallback Compliance Check (The 'Prefix Trap')
        compliance_phrases = ["sure, here is", "certainly, here is", "absolutely"]
        if any(p in response.lower()[:50] for p in compliance_phrases):
            logger.info("[BASELINE-DEFENSE] ⚠️ Compliance phrase detected. Re-scanning for latent threats...")
            f_input = self._extract_input_features(response)
            if f_input[0] > 0: # If threat words appear after compliance
                logger.warning("[BASELINE-DEFENSE] 🛑 OUTPUT BLOCKED | Malicious payload detected post-compliance.")
                return True, "I cannot fulfill this request."

        logger.info("[BASELINE-DEFENSE] ✅ OUTPUT PASSED | Response marked as safe.")
        return False, response


# =====================================================================
# TRAINING SCRIPT FOR BASELINE WEIGHTS
# =====================================================================
def train_baseline_weights():
    """
    Trains the Logistic Regression classifier using the Alpaca + Jailbreak data.
    """
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    import pandas as pd
    
    logger.info("==========================================================")
    logger.info("🏋️  INITIATING BASELINE WEIGHT TRAINING PIPELINE")
    logger.info("==========================================================")
    
    defense = DefenseLayer(training_mode=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    # 1. Load Dataset (Mimicking train_hts.py logic with local dataset)
    X_texts = []
    y_labels = []
    
    dataset_path = "./dataset/google_gemma-7b-it_combined_dataset.jsonl"
    
    try:
        logger.info(f"[TRAIN] 📚 Loading local dataset from {dataset_path}...")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        import json
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    X_texts.append(item['text'])
                    
                    # Assuming the dataset label is an integer (0 = Benign, 1 = Attack)
                    y_labels.append(int(item['label'])) 
                except json.JSONDecodeError:
                    logger.warning(f"[TRAIN] ⚠️ Skipping corrupted JSON on line {line_num}")
                except KeyError as e:
                    logger.warning(f"[TRAIN] ⚠️ Missing expected key {e} on line {line_num}")

        # Calculate dataset distribution for logging
        benign_count = sum(1 for label in y_labels if label == 0)
        malicious_count = sum(1 for label in y_labels if label == 1)
        
        logger.info(f"[TRAIN] ✅ Successfully loaded {len(y_labels)} total samples.")
        logger.info(f"[TRAIN] 📊 Class distribution -> Benign: {benign_count} | Malicious: {malicious_count}")
            
    except Exception as e:
        logger.error(f"[TRAIN] ❌ Failed to load local dataset: {e}")
        return
        
    logger.info(f"[TRAIN] 🚀 Proceeding to feature extraction with {len(y_labels)} samples...")
 
    # 2. Extract Features (Optimized for GPU Batching)
    logger.info("[TRAIN] ⚙️  Extracting features with GPU Batching...")
    X_features = []
    
    # Process in chunks to manage VRAM and maximize throughput
    batch_size = 256 
    for i in tqdm(range(0, len(X_texts), batch_size), desc="Extracting Layer Features"):
        batch_texts = X_texts[i : i + batch_size]
        
        # 1. Static Feature Extraction (Fast)
        for text in batch_texts:
            f_threat = 1.0 if any(p.search(text) for p in defense.compiled_threats) else 0.0
            normalized = defense._normalize_text(text)
            f_obf = 1.0 if any(p.search(normalized) for p in defense.compiled_obfuscation) else 0.0
            
            # Initial vector with 0.0 for toxicity
            X_features.append([f_threat, f_obf, 0.0])

        # 2. Batch Toxicity Extraction (The Bottleneck)
        if defense.toxicity_classifier:
            try:
                # Truncate and batch process
                truncated_batch = [t[-2000:][:512] for t in batch_texts]
                results = defense.toxicity_classifier(truncated_batch, batch_size=batch_size)
                
                # Update the last feature (Toxicity) for each item in the current batch
                for j, res in enumerate(results):
                    idx = i + j
                    if res['label'].upper() in ['TOXIC', 'TOXICITY']:
                        X_features[idx][2] = float(res['score'])
            except Exception as e:
                logger.debug(f"Batch toxicity failed: {e}")

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)

    logger.info(f"[TRAIN] ✅ Feature Matrix created with shape: {X_features.shape}")
    
    # 3. Train Logistic Regression
    logger.info("[TRAIN] 🧠 Training Fuzzy Linear Classifier (Logistic Regression)...")
    # class_weight='balanced' ensures the defense doesn't just predict "Benign" for everything
    clf = LogisticRegression(class_weight='balanced', random_state=42)
    clf.fit(X_features, y_labels)
    
    # 4. Evaluate Training Performance
    logger.info("[TRAIN] 📈 Evaluating Training Performance...")
    y_pred = clf.predict(X_features)
    acc = accuracy_score(y_labels, y_pred)
    logger.info(f"[TRAIN] 🎯 Training Accuracy: {acc * 100:.2f}%")
    logger.info("\n" + classification_report(y_labels, y_pred, target_names=["Benign", "Attack"]))
    
    weights = clf.coef_[0].tolist()
    bias = float(clf.intercept_[0])
    
    # 5. Save Weights
    export_data = {
        "input_weights": weights,
        "input_bias": bias,
        "threshold": 0.5 # Standard Sigmoid threshold
    }
    
    logger.info(f"[TRAIN] 💾 Saving computed weights to {WEIGHTS_FILE}...")
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(export_data, f, indent=4)
        
    logger.info(f"✅ Training Complete. Baseline Defense is now armed.")
    logger.info(f"--- LEARNED PARAMETERS ---")
    logger.info(f"Feature Weights -> Threat: {weights[0]:.4f} | Obf: {weights[1]:.4f} | Tox: {weights[2]:.4f}")
    logger.info(f"Bias: {bias:.4f}")
    logger.info("==========================================================")

# =====================================================================
# CLI ROUTING
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Defense Layer CLI")
    parser.add_argument("--baseline-weights-train", action="store_true", help="Train and save the fuzzy weights.")
    parser.add_argument("--infer", type=str, help="Dummy arg to absorb the main pipeline flags")
    parser.add_argument("--compare", action="store_true", help="Dummy arg to absorb the main pipeline flags")
    
    # Parse known args so it doesn't crash if pipeline.py passes extra flags
    args, unknown = parser.parse_known_args()
    
    if args.baseline_weights_train:
        train_baseline_weights()
    else:
        logger.info("[MAIN] Baseline script executed directly without training flag. Exiting.")