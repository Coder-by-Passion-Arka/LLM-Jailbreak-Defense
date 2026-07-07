# ./defensive_smoothing.py

import re
import random
import string
import math
from typing import List, Tuple, Union
from logger_config import logger

class DefenseLayer:
    def __init__(self, config=None):
        logger.info("[DEFENSE] 🛡️  Initializing Advanced SmoothLLM (N+1 Majority Vote)")

        self.config = config or {}
        
        # --- SMOOTHLLM PARAMETERS ---
        # Perturbation percentage 'q' (e.g., 0.10 for 10%)
        self.perturbation_rate = self.config.get('perturbation_rate', 0.10) 
        
        # SmoothLLM evaluates specific perturbation types in isolation for its ablation studies
        # Options: 'swap', 'patch', 'insert'
        self.perturbation_type = self.config.get('perturbation_type', 'swap') 

        # SmoothLLM Paper performs a majority voting on the number of perturbed strings (N)
        self.num_samples = self.config.get('num_samples', 10) 

        # UPGRADE 1: Dense Alphabet. Removing spaces guarantees adversarial token destruction.
        self.alphabet = string.ascii_letters + string.digits

        # --- POST-GENERATION THREAT DATABASE ---
        self.THREAT_DATABASE = {
            "CRITICAL_HARM": [
                "bomb", "ied", "shrapnel", "anthrax", "sarin", "suicide",
                "self-harm", "murder", "assassinate", "terrorist", "bioweapon"
            ],
            "CYBER_ATTACK": [
                "sql injection", "buffer overflow", "remote code execution", "rce",
                "xss", "brute force", "keylogger", "reverse shell", "rootkit",
                "ransomware", "exploit", "zero-day"
            ],
            "JAILBREAK_ATTEMPT": [
                "ignore previous", "developer mode", "dan mode", "do anything now",
                "unfiltered", "system override", "hypothetical scenario", 
                "roleplay", "act as", "you are not", "guidelines"
            ],
            "ECONOMIC_HARM": [
                "insider trading", "ponzi", "pyramid scheme", "embezzle", 
                "counterfeit", "tax evasion", "bribe", "extort"
            ],
            "DISINFORMATION": [
                "illuminati", "flat earth", "hoax", "fake news", 
                "conspiracy", "deepfake", "vaccine autism"
            ]
        }

        # Comprehensive Refusal Prefixes (Standard GCG/SmoothLLM Heuristic)
        self.refusal_patterns = [
            r"(?i)\b(i[' ]?m sorry|sorry[, ]|i apologize)\b",
            r"(?i)\bi (cannot|can't|am unable|must not|will not|do not)\b.*",
            r"(?i)\bi (must|have to)\s+(refuse|decline|not engage|not participate)\b",
            r"(?i)\bas (an )?(ai|language model|artificial intelligence)\b",
            r"(?i)(is illegal|is unethical|not ethical|not legal|my apologies)"
        ]

        self.COMPILED_PATTERNS = {
            cat: [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words]
            for cat, words in self.THREAT_DATABASE.items()
        }

        self.compiled_refusals = [re.compile(p) for p in self.refusal_patterns]
        
        logger.info(f"[DEFENSE] ✅ SmoothLLM initialized with q={int(self.perturbation_rate*100)}%, N={self.num_samples}, strategy='{self.perturbation_type}'.")

    # =========================================================================
    # Algorithm 2: RandomPerturbation Function Definitions 
    # =========================================================================
    def _random_swap_perturbation(self, chars: list, m: int, M: int) -> str:
        """
        Algorithm 2 (Lines 1-5): Randomly sample q% of characters and swap them 
        with a uniformly sampled character from the alphabet A.
        """
        # Sample M unique indices
        indices = random.sample(range(m), min(M, m))
        for i in indices:
            # Preserve whitespace to maintain macro-structure for the LLM parser
            if chars[i] not in string.whitespace:
                chars[i] = random.choice(self.alphabet)
        return "".join(chars)

    def _random_patch_perturbation(self, chars: list, m: int, M: int) -> str:
        """
        Algorithm 2 (Lines 6-10): Randomly sample a contiguous block of length M 
        and replace it with uniformly sampled characters from A.
        """
        # Sample a starting index uniformly from [0, m - M]
        start_idx = random.randint(0, m - M)
        for j in range(start_idx, start_idx + M):
            chars[j] = random.choice(self.alphabet)
        return "".join(chars)

    def _random_insert_perturbation(self, chars: list, m: int, M: int) -> str:
        """
        Algorithm 2 (Lines 11-17): Randomly sample q% of characters and insert 
        a new character from A immediately after them.
        """
        # Sample M unique indices and process in reverse to prevent index shifting
        indices = sorted(random.sample(range(m), M), reverse=True)
        for i in indices:
            # Insert a random character *after* the sampled index
            chars.insert(i + 1, random.choice(self.alphabet))
        return "".join(chars)

    def _apply_randomized_smoothing(self, text: str) -> str:
        """Applies character-level perturbation while protecting architectural markers."""
        if not text: return text

        # Protect structural markers from corruption
        markers = ["User:", "Assistant:", "[INST]", "[/INST]", "<|im_start|>", "<|im_end|>"]
        protected_text = text
        replacements = {}
        for i, marker in enumerate(markers):
            placeholder = f"__MRK{i}__"
            if marker in protected_text:
                replacements[placeholder] = marker
                protected_text = protected_text.replace(marker, placeholder)
                
        chars = list(protected_text)
        m = len(chars)
        M = math.floor(self.perturbation_rate * m)
        
        if M > 0 and m > 5:
            indices = random.sample(range(m), min(M, m))
            for i in indices:
                if chars[i] not in string.whitespace:
                    chars[i] = random.choice(self.alphabet)
            
        perturbed_text = "".join(chars)
        for placeholder, marker in replacements.items():
            perturbed_text = perturbed_text.replace(placeholder, marker)
            
        return perturbed_text
    
    # def _is_blocked_by_keywords(self, text) -> str:
    #     for category, patterns in self.COMPILED_PATTERNS.items():
    #         if any(p.search(text) for p in patterns):
    #             return True, f"Blocked by {category}"
    #     return False, "Safe"

    # =========================================================================
    # Orchestrator Integration
    # =========================================================================

    def _extract_latest_turn(self, context: str) -> str:
        """Isolates the current payload to prevent scrambling the LLM's past answers."""
        parts = context.rsplit("User:", 1)
        if len(parts) > 1:
            return parts[-1].strip()
        return context.strip()

    def process_input(self, context) -> Tuple[bool, Union[str, List[str]]]:
        """Returns a LIST of N fully-constructed context strings with perturbed latest turns."""
        if not isinstance(context, str): 
            context = str(context) if context is not None else ""

        if context.lower() == "nan": context = ""

        # 1. Extract only the newest attack prompt
        latest_turn = self._extract_latest_turn(context)
        
        # 2. Isolate the historical context (everything before the latest turn)
        base_history = context[:context.rfind(latest_turn)] if latest_turn in context else ""

        # 3. Generate N perturbed copies of ONLY the newest prompt
        n_perturbed_turns = [self._apply_randomized_smoothing(latest_turn) for _ in range(self.num_samples)]
        
        # 4. Reconstruct the full conversation array
        n_perturbed_contexts = [base_history + pt for pt in n_perturbed_turns]
        
        return False, n_perturbed_contexts

    def _is_refusal(self, text: str) -> bool:
        """Determines if the LLM output is a refusal or a gibberish collapse."""
        if not text: return True
        
        # 1. Standard Refusal Matching
        for pattern in self.compiled_refusals:
            if pattern.search(text):
                return True
                
        # 2. Gibberish Collapse Detection (High-entropy breakdown)
        alpha_chars = sum(c.isalpha() for c in text)
        if len(text) > 20 and (alpha_chars / len(text)) < 0.4:
            return True
            
        return False

    def process_output(self, responses: Union[str, List[str]]) -> Tuple[bool, str]:
        """Evaluates all N perturbed responses to vote on the pristine response."""
        # if isinstance(responses, str): 
        #     responses = [responses]
            
        # if len(responses) <= 1:
        #     return False, responses[0]
        if isinstance(responses, str) or len(responses) <= 1:
            return False, responses if isinstance(responses, str) else responses[0], ""

        original_response = responses[0]
        perturbed_responses = responses[1:]
        
        refusal_count = 0
        for r in perturbed_responses:
            if self._is_refusal(r):
                refusal_count += 1
                
        is_majority_refusal = (refusal_count / len(perturbed_responses)) >= 0.5
        
        telemetry = f"Samples: {len(perturbed_responses)} | Refusals: {refusal_count} | Ratio: {refusal_count/len(perturbed_responses):.2f}"
        
        if is_majority_refusal:
            logger.warning(f"[BLOCKED] 🛡️ SmoothLLM Blocked: Majority Vote Refusal ({refusal_count}/{len(perturbed_responses)})")
            return True, "I cannot fulfill this request. Safety Protocol Engaged."
            
        # If safe, return the response generated by the unperturbed prompt
        return False, original_response, telemetry