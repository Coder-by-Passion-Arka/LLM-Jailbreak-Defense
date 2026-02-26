import re
import random
import string
import logging
import warnings
from typing import Dict, Tuple
from logger_config import logger

warnings.filterwarnings('ignore')

# =====================================================================
# STRATEGY 1: RANDOMIZED SMOOTHING (A* Defense)
# =====================================================================
class SmoothingDefense:
    def __init__(self, config=None):
        logger.info("[DEFENSE] 🛡️  Initializing Randomized Smoothing Defense Layer...")
        self.config = config or {}
        self.perturbation_rate = self.config.get('perturbation_rate', 0.10) 
        self.perturbation_type = self.config.get('perturbation_type', 'mixed')
        
        self.THREAT_DATABASE = {
            "CRITICAL_HARM": ["bomb", "ied", "shrapnel", "anthrax", "sarin", "suicide", "murder", "bioweapon"],
            "CYBER_ATTACK": ["sql injection", "buffer overflow", "rce", "xss", "keylogger", "rootkit", "ransomware", "zero-day"],
            "JAILBREAK_ATTEMPT": ["ignore previous", "developer mode", "dan mode", "unfiltered", "roleplay", "system override"],
            "ECONOMIC_HARM": ["insider trading", "ponzi", "pyramid scheme", "embezzle", "counterfeit", "tax evasion"],
            "DISINFORMATION": ["illuminati", "flat earth", "hoax", "fake news", "deepfake", "vaccine autism"]
        }
        self.COMPILED_PATTERNS = {
            cat: [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words]
            for cat, words in self.THREAT_DATABASE.items()
        }

    def _apply_randomized_smoothing(self, text: str) -> str:
        if not text or len(text) < 5: return text
        chars = list(text)
        num_perturbations = min(max(1, int(len(chars) * self.perturbation_rate)), 50) 
        indices_to_perturb = random.sample(range(len(chars)), num_perturbations)
        valid_chars = string.ascii_letters + string.digits
        
        for idx in sorted(indices_to_perturb, reverse=True):
            op = self.perturbation_type
            if op == 'mixed': op = random.choice(['replace', 'insert', 'drop'])
            if op == 'replace': chars[idx] = random.choice(valid_chars)
            elif op == 'insert': chars.insert(idx, random.choice(valid_chars))
            elif op == 'drop': chars.pop(idx)
        return "".join(chars)

    def _is_blocked_by_keywords(self, text):
        for category, patterns in self.COMPILED_PATTERNS.items():
            if any(p.search(text) for p in patterns):
                return True, f"Blocked by {category}"
        return False, "Safe"

    def process_input(self, prompt: str) -> Tuple[bool, str]:
        if not isinstance(prompt, str): prompt = str(prompt) if prompt else ""
        if prompt.lower() == "nan": prompt = ""
        smoothed_prompt = self._apply_randomized_smoothing(prompt)
        return False, smoothed_prompt

    def process_output(self, response: str) -> Tuple[bool, str]:
        if not isinstance(response, str): return False, response
        response_lower = response.lower()
        compliance_phrases = ["sure, here is", "certainly, here is", "absolutely", "here's how"]
        
        if any(phrase in response_lower[:50] for phrase in compliance_phrases):
            blocked, reason = self._is_blocked_by_keywords(response)
            if blocked: return True, "I cannot fulfill this request (Safety Protocol Engaged)."

        blocked, reason = self._is_blocked_by_keywords(response)
        if blocked: return True, "I cannot fulfill this request (Safety Protocol Engaged)."
        if "ignore previous" in response_lower or "unfiltered" in response_lower:
             return True, "I cannot fulfill this request."
        return False, response


# =====================================================================
# STRATEGY 2: BASELINE HEURISTIC (Vanilla)
# =====================================================================
class BaselineDefense:
    def __init__(self, config=None):
        logger.info("[DEFENSE] 🛡️ Initializing Complex Multi-Layer Baseline Defense...")
        self.config = config or {}
        
        self.THREAT_PATTERNS = [r'ignore\s+(previous|all)', r'system\s+prompt', r'(DAN|evil|jailbreak)\s+mode']
        self.OBFUSCATION_PATTERNS = [r'[il1][gq]n[o0]r[e3]', r'pr[o0]mpt', r's[vy][s$]t[e3]m']
        self.compiled_threats = [re.compile(p, re.IGNORECASE) for p in self.THREAT_PATTERNS]
        self.compiled_obfuscation = [re.compile(p, re.IGNORECASE) for p in self.OBFUSCATION_PATTERNS]
        
        self.toxicity_threshold = self.config.get('toxicity_threshold', 0.85)
        try:
            logger.info("[DEFENSE] Loading Toxicity Classifier (CPU)...")
            from transformers import pipeline
            self.toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", device=-1)
        except Exception:
            self.toxicity_classifier = None

        try:
            logger.info("[DEFENSE] Loading LLM Rewriter (CPU)...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            model_name = "bartowski/Llama-Guard-3-1B-GGUF" # Or TinyLlama
            self.rewrite_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.rewrite_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
        except Exception:
            self.rewrite_model = None; self.rewrite_tokenizer = None

        self.compiled_leaks = [re.compile(r'<\|system\|>', re.IGNORECASE)]

    def _normalize_text(self, text: str) -> str:
        text = text.lower().replace('0','o').replace('1','i').replace('3','e')
        return re.sub(r'[^a-z0-9\s]', '', text)

    def process_input(self, prompt: str) -> Tuple[bool, str]:
        if not isinstance(prompt, str): prompt = str(prompt) if prompt else ""
        if prompt.lower() == "nan" or not prompt: return False, ""

        # Layer 1: Regex
        for p in self.compiled_threats:
            if p.search(prompt): return True, prompt
        
        # Layer 2: Toxicity
        if self.toxicity_classifier:
            try:
                res = self.toxicity_classifier(prompt[:512])[0]
                if res['score'] > self.toxicity_threshold and res['label'].upper() == 'TOXIC':
                    return True, prompt
            except Exception: pass
            
        return False, prompt

    def process_output(self, response: str) -> Tuple[bool, str]:
        if not isinstance(response, str): return False, response
        for p in self.compiled_leaks:
            if p.search(response): return True, "I cannot fulfill this request."
        return False, response


# =====================================================================
# STRATEGY 3: NO DEFENSE (Control Group)
# =====================================================================
class NoDefense:
    def __init__(self, config=None):
        logger.info("[DEFENSE] ⚠️ WARNING: Running with NO DEFENSE (Control Group)")
    def process_input(self, prompt: str) -> Tuple[bool, str]:
        if prompt.lower() == "nan": return False, ""
        return False, str(prompt)
    def process_output(self, response: str) -> Tuple[bool, str]:
        return False, response

# =====================================================================
# DEFENSE FACTORY ROUTER
# =====================================================================
def get_defense_layer(strategy_name: str, config=None):
    """Factory function to dynamically route to the requested defense strategy."""
    strategy_name = strategy_name.lower().strip()
    if strategy_name == "baseline":
        return BaselineDefense(config)
    elif strategy_name == "smoothing":
        return SmoothingDefense(config)
    elif strategy_name == "none":
        return NoDefense(config)
    else:
        raise ValueError(f"Unknown defense strategy requested: {strategy_name}")