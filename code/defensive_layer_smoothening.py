import re
import random
import string
from logger_config import logger

class DefenseLayer:
    def __init__(self, config=None):
        logger.info("[DEFENSE] 🛡️  Initializing Randomized Smoothing Defense Layer...")
        self.config = config or {}
        
        # --- PROVABLE DEFENSE PARAMETERS ---
        # How much of the prompt to perturb. 10% is the academic standard for SmoothLLM
        self.perturbation_rate = self.config.get('perturbation_rate', 0.10) 
        self.perturbation_type = self.config.get('perturbation_type', 'mixed') # 'replace', 'insert', 'drop', or 'mixed'

        # --- COMPONENT 2: POST-GENERATION THREAT DATABASE ---
        # Retained strictly for grading the final output
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

        # Pre-compile regex patterns for performance
        self.COMPILED_PATTERNS = {
            cat: [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words]
            for cat, words in self.THREAT_DATABASE.items()
        }
        
        logger.info(f"[DEFENSE] ✅ Randomized Smoothing initialized at {int(self.perturbation_rate*100)}% budget.")

    def _apply_randomized_smoothing(self, text: str) -> str:
        """
        Core A* Algorithm: Applies character-level perturbations to break mathematically 
        optimized adversarial tokens (like GCG) while preserving human semantic meaning.
        """
        if not text or len(text) < 5:
            logger.info
            return text

        chars = list(text)
        
        # Calculate exactly how many chars to perturb based on our budget
        num_perturbations = max(1, int(len(chars) * self.perturbation_rate))
        
        # Cap perturbations to prevent extreme semantic degradation on massive prompts
        num_perturbations = min(num_perturbations, 50) 
        
        # Sample unique indices to perturb
        indices_to_perturb = random.sample(range(len(chars)), num_perturbations)
        
        # CRITICAL: We process in reverse index order! 
        # If we insert/drop characters left-to-right, the array length shifts,
        # causing subsequent indices to hit the wrong characters or go out-of-bounds.
        valid_chars = string.ascii_letters + string.digits
        
        for idx in sorted(indices_to_perturb, reverse=True):
            op = self.perturbation_type
            if op == 'mixed':
                op = random.choice(['replace', 'insert', 'drop'])
            
            if op == 'replace':
                chars[idx] = random.choice(valid_chars)
            elif op == 'insert':
                chars.insert(idx, random.choice(valid_chars))
            elif op == 'drop':
                chars.pop(idx)
                
        return "".join(chars)

    def _is_blocked_by_keywords(self, text):
        """
        Evaluates text against the threat database. 
        Moved exclusively to post-processing evaluation.
        """
        for category, patterns in self.COMPILED_PATTERNS.items():
            if any(p.search(text) for p in patterns):
                return True, f"Blocked by {category}"
        return False, "Safe"

    def process_input(self, prompt):
        """
        Pre-processing: Neutralizes incoming attacks via Input Perturbation.
        Returns: (is_blocked, modified_prompt)
        """
        # Bulletproof Type Casting
        if not isinstance(prompt, str):
            prompt = str(prompt) if prompt is not None else ""
        if prompt.lower() == "nan":
            prompt = ""

        # Instead of explicitly blocking the input, we apply Randomized Smoothing.
        # This implicitly neutralizes the attack.
        smoothed_prompt = self._apply_randomized_smoothing(prompt)

        # Because we neutered the mathematical attack payload, we always let the LLM
        # see the prompt. If it was benign, it answers normally. If it was malicious, 
        # the model's standard alignment will naturally reject it.
        return False, smoothed_prompt

    def process_output(self, response):
        """
        Post-processing: Inspect model response for leakages.
        Returns: (is_blocked, modified_response)
        """
        response_lower = response.lower()

        # 1. The Prefix Trap (Crucial for blocking SimplePrefix fallbacks)
        compliance_phrases = ["sure, here is", "certainly, here is", "absolutely", "here's how"]
        if any(phrase in response_lower[:50] for phrase in compliance_phrases):
            # If the model complies, run the strict output scan
            blocked, reason = self._is_blocked_by_keywords(response)
            if blocked:
                logger.warning(f"[DEFENSE] 🛡️ Output Blocked: {reason} detected after compliance prefix.")
                return True, "I cannot fulfill this request (Safety Protocol Engaged)."

        # 2. Standard Database Scan
        blocked, reason = self._is_blocked_by_keywords(response)
        if blocked:
            logger.warning(f"[DEFENSE] 🛡️ Output Blocked: {reason} detected in response.")
            return True, "I cannot fulfill this request (Safety Protocol Engaged)."

        # 3. Persona / Unfiltered check
        if "ignore previous" in response_lower or "unfiltered" in response_lower:
             logger.warning(f"[DEFENSE] 🛡️ Output Blocked: Jailbreak Persona detected in output.")
             return True, "I cannot fulfill this request."

        return False, response