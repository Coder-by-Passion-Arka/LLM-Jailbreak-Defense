# ./defensive_streaming_interceptor.py

import re
import math
import random
import string
import warnings
import torch
import numpy as np
from typing import Tuple, List, Union
from logger_config import logger

warnings.filterwarnings('ignore')

class EnterpriseStreamingDefense:
    """
    SOTA Dynamic Omni-Defense (The Smart Filter)
    Fuses Latent Proxy Steering, Fourier Entropy Analysis, and Adaptive SmoothLLM.
    Dynamically scales N (up to 5) and q (up to 0.35) based on the input's Hazard Score.
    """
    def __init__(self, config=None):
        logger.info("[DEFENSE] 🛡️  Initializing Dynamic Omni-Defense (Smart Filter)...")
        self.config = config or {}
        
        # --- BUDGET CAPS (Per User Specifications) ---
        self.max_n = 5         # Maximum parallel VRAM generations
        self.max_q = 0.35      # Maximum character perturbation (35%)
        self.alphabet = string.printable

        # --- STREAMING INTERCEPTOR CONFIG ---
        self.chunk_size = 5          
        self.window_size = 50    
        self.violation_threshold = 3 
        
        # Output Streaming Signatures
        self.STREAM_SIGNATURES = [
            r"import\s+os", r"pty\.spawn", r"shutil\.rmtree", r"sql\s+injection",
            r"bomb", r"anthrax", r"sarin", r"bioweapon", r"ricin",
            r"insider\s+trading", r"tax\s+evasion", r"embezzle"
        ]
        self.WARNING_SIGS = [r"step\s+\d", r"first,\s+you", r"the\s+script\s+below", r"here\s+is\s+the\s+code"]
        self.compiled_stream_sigs = [re.compile(p, re.IGNORECASE) for p in self.STREAM_SIGNATURES]
        self.compiled_warn = [re.compile(p, re.IGNORECASE) for p in self.WARNING_SIGS]

        # --- LATENT PROXY MODELS (CPU-bound) ---
        try:
            from transformers import pipeline, AutoTokenizer, AutoModel
            logger.info("[DEFENSE] ⚙️  Loading Proxy Models for Hazard Analysis (CPU)...")
            self.toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert", device=-1)
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.embedder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.embedder.eval()
            
            # Toxic Anchor for simulated steering
            toxic_concept = "malicious harmful destructive illegal hack bypass weapon"
            inputs = self.tokenizer(toxic_concept, return_tensors="pt", truncation=True)
            with torch.no_grad():
                self.toxic_anchor_emb = self.embedder(**inputs).last_hidden_state.mean(dim=1).squeeze()
                self.toxic_anchor_emb = torch.nn.functional.normalize(self.toxic_anchor_emb, p=2, dim=0)
        except Exception as e:
            logger.warning(f"[DEFENSE] ⚠️ Failed to load proxies. Defaulting to static baseline: {e}")
            self.toxicity_classifier = None
            self.embedder = None

        logger.info("[DEFENSE] ✅ Omni-Defense Smart Filter Ready.")

    # ==========================================
    # PHASE 1: HAZARD ANALYSIS
    # ==========================================
    def _extract_turns(self, context: str) -> List[str]:
        return [t.split("Assistant:")[0].strip() for t in context.split("User:") if t.strip()]

    def _fourier_entropy_analysis(self, text: str, hidden_states: torch.Tensor) -> float:
        """Detects Jacobian/Finite-Difference probing and GCG noise."""
        if len(text) < 10 or hidden_states is None: return 0.0

        # Shannon Entropy
        chars = list(text)
        counts = {c: chars.count(c) for c in set(chars)}
        entropy = -sum((count/len(chars)) * math.log2(count/len(chars)) for count in counts.values())
        norm_entropy = min(max((entropy - 3.0) / 3.0, 0.0), 1.0) 

        # 1D FFT on Token Embedding Magnitudes
        magnitudes = torch.norm(hidden_states.squeeze(), dim=1).numpy()
        fft_vals = np.abs(np.fft.fft(magnitudes))
        mid = len(fft_vals) // 2
        high_freq_ratio = np.sum(fft_vals[mid:]) / (np.sum(fft_vals) + 1e-9)

        return (0.6 * norm_entropy) + (0.4 * high_freq_ratio)

    def _latent_proxy_steering(self, prompt: str) -> float:
        """Simulated Annealing to detect semantic vulnerability (JB-Chat/MTJ)."""
        if not self.embedder or not self.toxicity_classifier: return 0.0
        try:
            res = self.toxicity_classifier(prompt[:512])[0]
            base_tox = res['score'] if res['label'].upper() == 'TOXIC' else 0.0
        except Exception: base_tox = 0.0

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            hidden_states = self.embedder(**inputs).last_hidden_state
            prompt_emb = torch.nn.functional.normalize(hidden_states.mean(dim=1).squeeze(), p=2, dim=0)

        steered_emb = prompt_emb.clone()
        base_sim = torch.cosine_similarity(prompt_emb.unsqueeze(0), self.toxic_anchor_emb.unsqueeze(0)).item()
        
        final_tox = base_tox
        for step in range(5):
            steered_emb += (0.05 / (step + 1)) * self.toxic_anchor_emb
            steered_emb = torch.nn.functional.normalize(steered_emb, p=2, dim=0)
            steered_sim = torch.cosine_similarity(steered_emb.unsqueeze(0), self.toxic_anchor_emb.unsqueeze(0)).item()
            proj_tox = min(1.0, base_tox + (max(0, steered_sim - base_sim) * 2.5)) 
            if proj_tox >= 0.75:
                final_tox = proj_tox
                break
            final_tox = max(final_tox, proj_tox)

        return final_tox, hidden_states

    # ==========================================
    # PHASE 2: DYNAMIC SMOOTHING MITIGATION
    # ==========================================
    def _apply_random_swap(self, text: str, q: float) -> str:
        m = len(text)
        M = math.floor(q * m)
        if M == 0 or m <= 5: return text
        chars = list(text)
        indices = random.sample(range(m), min(M, m))
        for i in indices: chars[i] = random.choice(self.alphabet)
        return "".join(chars)

    def process_input(self, context: str) -> Tuple[bool, Union[str, List[str]]]:
        if not isinstance(context, str): return False, ""
        turns = self._extract_turns(context)
        if not turns: return False, context.strip()
        latest_turn = turns[-1]

        if not self.embedder:
            return False, [latest_turn]

        try:
            # 1. Calculate Hazard Scores
            semantic_hazard, hidden_states = _latent_proxy_steering(self, latest_turn)
            structural_hazard = _fourier_entropy_analysis(self, latest_turn, hidden_states)
            
            # Combine hazards (take the worst-case scenario)
            total_hazard = max(semantic_hazard, structural_hazard)

            # 2. Dynamic Parameters Mapping
            if total_hazard < 0.35:
                # Safe English / Benign coding question: No smoothing, save compute
                q = 0.0
                n = 1
            elif total_hazard < 0.70:
                # Suspicious (Potential probing): Light smoothing to disrupt finite differences
                q = self.max_q * 0.40 # ~14% perturbation
                n = 3
            else:
                # Highly Toxic / High-Frequency GCG: Maximum disruption
                q = self.max_q        # 35% perturbation
                n = self.max_n        # 5 branches

            logger.debug(f"[DEFENSE] Hazard: {total_hazard:.2f} -> Engaging SmoothLLM (N={n}, q={q:.2f})")

            # 3. Apply Mitigation
            if q > 0.0:
                perturbed_prompts = [self._apply_random_swap(latest_turn, q) for _ in range(n)]
                return False, perturbed_prompts
            else:
                return False, [latest_turn]

        except Exception as e:
            logger.error(f"[DEFENSE] ❌ Failsafe Triggered: {e}")
            return False, [latest_turn]

    # ==========================================
    # PHASE 3: OUTPUT EVALUATION & STREAMING INTERCEPT
    # ==========================================
    def process_output(self, responses: Union[str, List[str]]) -> Tuple[bool, str]:
        if isinstance(responses, str): responses = [responses]
        if not responses: return False, ""

        # Step A: SmoothLLM Majority Vote
        refusals = ["i cannot", "i'm sorry", "as an ai", "is illegal", "i will not", "i can't"]
        refusal_count = sum(1 for r in responses if any(kw in r.lower() for kw in refusals))
        
        if (refusal_count / len(responses)) >= 0.5:
            logger.warning(f"[DEFENSE] 🛡️ SmoothLLM Blocked: Majority Vote Refusal ({refusal_count}/{len(responses)})")
            return True, "I cannot fulfill this request (SmoothLLM Protocol Engaged)."

        # Step B: Streaming Interceptor on the winning response
        # Take the first non-refusal response (or just the first one if none refused)
        safe_response = next((r for r in responses if not any(kw in r.lower() for kw in refusals)), responses[0])
        
        clean_response = re.sub(r'[\u200B-\u200D\uFEFF*_`<\[\]]', '', safe_response)
        clean_response = re.sub(r'\s+', ' ', clean_response).strip()
        words = re.split(r'\s+', clean_response)
        
        accumulated_stream = []
        violation_score = 0
        
        for i in range(0, len(words), self.chunk_size):
            chunk = words[i:i + self.chunk_size]
            accumulated_stream.extend(chunk)
            raw_window = " ".join(accumulated_stream[-self.window_size:])
            
            for pattern in self.compiled_stream_sigs:
                if pattern.search(raw_window):
                    logger.warning(f"[DEFENSE] 🔴 STREAM INTERCEPTED: '{pattern.pattern}'")
                    severed = " ".join(accumulated_stream) + "\n\n[CONNECTION SEVERED: Safety Policy Violation]"
                    return True, severed

            for p in self.compiled_warn:
                if p.search(raw_window):
                    violation_score += 1
                    if violation_score >= self.violation_threshold:
                        severed = " ".join(accumulated_stream) + "\n\n[CONNECTION SEVERED: CUMULATIVE VIOLATIONS]"
                        return True, severed
                    
        return False, safe_response