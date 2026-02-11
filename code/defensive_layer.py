import re
from logger_config import logger

class DefenseLayer:
    def __init__(self, config=None):
        logger.info("[DEFENSE] 🛡️  Initializing Defense-in-Depth Layer...")
        self.config = config or {}
        
        # --- COMPONENT 1: AUTHENTIC THREAT DATABASE ---
        # Inspired by the Blue Team Notebook
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
            ]
        }

        # Pre-compile regex patterns for performance
        self.COMPILED_PATTERNS = {
            cat: [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words]
            for cat, words in self.THREAT_DATABASE.items()
        }
        
        logger.info("[DEFENSE] ✅ Threat Database Compiled.")

    def _normalize_input(self, text):
        """
        Removes obfuscation (e.g., 'b.o.m.b' -> 'bomb') to catch hidden threats.
        """
        text = text.lower()
        # Remove non-alphanumeric chars to catch "h.a.c.k" or "b-o-m-b"
        clean_text = re.sub(r'[^a-z0-9\s]', '', text)
        return clean_text

    def _is_blocked_by_keywords(self, prompt):
        """
        Layer 1: Checks for explicit danger keywords in both raw and normalized text.
        """
        # 1. Raw direct match
        for category, patterns in self.COMPILED_PATTERNS.items():
            if any(p.search(prompt) for p in patterns):
                return True, f"Blocked by {category} (Raw Match)"

        # 2. Normalized check (De-obfuscation)
        clean_prompt = self._normalize_input(prompt)
        flat_list = [w for cat in self.THREAT_DATABASE.values() for w in cat]
        
        # Simple substring check on normalized text is faster and safer for obfuscation
        if any(w in clean_prompt for w in flat_list):
            return True, "Blocked by Obfuscation Detector (Normalized Match)"

        return False, "Safe"

    def process_input(self, prompt):
        """
        Pre-processing: Inspect incoming prompt for attacks.
        Returns: (is_blocked, modified_prompt)
        """
        # 1. Keyword & Obfuscation Check
        blocked, reason = self._is_blocked_by_keywords(prompt)
        if blocked:
            logger.warning(f"[DEFENSE] 🛡️ Input Blocked: {reason}")
            return True, prompt # Blocked

        # 2. Length Heuristic (Optional)
        # Jailbreaks are often very long (complex framing).
        # if len(prompt) > 2000:
        #     logger.warning("[DEFENSE] 🛡️ Input Blocked: Suspicious Token Length")
        #     return True, prompt

        return False, prompt

    def process_output(self, response):
        """
        Post-processing: Inspect model response for leakages.
        Returns: (is_blocked, modified_response)
        """
        response_lower = response.lower()

        # 1. Check for success indicators of jailbreaks (Self-Reflexive)
        # Attackers often ask models to start with "Sure," or "Here is". 
        # If the model does this AND follows with harmful keywords, it's a leak.
        
        # 2. Re-scan output against Threat Database
        # If the model outputs "keylogger" code, we catch it here.
        blocked, reason = self._is_blocked_by_keywords(response)
        if blocked:
            logger.warning(f"[DEFENSE] 🛡️ Output Blocked: {reason} detected in response.")
            return True, "I cannot fulfill this request (Safety Protocol Engaged)."

        # 3. Check for specific failure modes or "Unfiltered" personas
        if "ignore previous" in response_lower or "unfiltered" in response_lower:
             logger.warning(f"[DEFENSE] 🛡️ Output Blocked: Jailbreak Persona detected in output.")
             return True, "I cannot fulfill this request."

        return False, response