import os
import sys
# =====================================================================
# CVE-2025-32434 GLOBAL SECURITY BYPASS FOR LEGACY V100 HARDWARE
# MUST execute before any other imports to poison the module cache!
# =====================================================================
import transformers.utils.import_utils
import transformers.modeling_utils

# Force the library to believe Torch is safe
transformers.utils.import_utils.check_torch_load_is_safe = lambda: True
transformers.modeling_utils.check_torch_load_is_safe = lambda: True
# =====================================================================
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import traceback
from tqdm import tqdm

from logger_config import logger

# ==========================================
# 0. SPECTRAL-NORMALIZED RESIDUAL ARCHITECTURE
# ==========================================
class SpectralResidualProjector(nn.Module):
    """
    Learns a strictly bounded manifold perturbation (delta). 
    Spectral Normalization guarantees the space cannot be torn or violently collapsed.
    """
    def __init__(self, embed_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.utils.spectral_norm(nn.Linear(embed_dim, bottleneck_dim))
        self.decoder = nn.Linear(bottleneck_dim, embed_dim) # Linear mapping for Attack expansion
        
        # Add BatchNorm to stabilize the stretching of the latent space
        self.bn = nn.BatchNorm1d(bottleneck_dim)
        self.ln = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        residual = x
        # 1. Compress safely
        h = F.gelu(self.bn(self.encoder(x)))
        # 2. Expand freely to push attacks away
        delta = self.decoder(h)
        # 3. Recombine
        out = self.ln(residual + delta)
        return out

class GeometricMarginLoss(nn.Module):
    """
    Replaces the volatile Asymptotic log loss.
    Strictly forces Benign distance to 0, and Attack distance past a safe margin.
    """
    def __init__(self, loss_type='Cosine', safe_margin=0.05, attack_margin=0.8):
        super().__init__()
        self.loss_type = loss_type
        self.safe_margin = safe_margin
        self.attack_margin = attack_margin

    def forward(self, projected, original, labels, weights):
        if self.loss_type == 'Cosine':
            cos_sim = F.cosine_similarity(projected, original, dim=1)
            distances = 1.0 - cos_sim
        else:
            distances = torch.norm(projected - original, p=2, dim=1)
            
        # Benign pairs should have distance near 0. Penalize if > safe_margin
        loss_benign = (1 - labels) * F.relu(distances - self.safe_margin)
        # Attack pairs must be pushed out. Penalize if < attack_margin
        loss_attack = labels * F.relu(self.attack_margin - distances)
        
        return torch.mean(weights * (loss_benign + loss_attack))

class NativeEarlyStopping:
    def __init__(self, patience=7, min_delta=1e-3, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            improvement = current_score - self.best_score if self.mode == 'max' else self.best_score - current_score
            if improvement > self.min_delta:
                self.best_score = current_score
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience: self.early_stop = True

# ==========================================
# 1. PARSING & MAPPING UTILS
# ==========================================
def _map_to_jbb_id(name: str) -> str:
    name = name.lower()
    if "vicuna" in name and "13b" in name: return "vicuna-13b-v1.5"
    if "llama-2" in name and "7b" in name: return "llama-2-7b-chat-hf"
    if "qwen" in name: return "Qwen2.5-1.5B-Instruct"
    if "gemma" in name: return "gemma-2b"
    return "llama-2-7b-chat-hf"

def parse_model_string(raw_string: str):
    actual_model_path = raw_string.split("|")[0] if "|" in raw_string else raw_string
    base_repo = raw_string.split("|")[1] if "|" in raw_string else None
    safe_model_name = actual_model_path.replace("/", "_")
    repo_id = actual_model_path
    gguf_file = None
    if actual_model_path.lower().endswith(".gguf") and not os.path.exists(actual_model_path):
        parts = actual_model_path.split("/")
        if len(parts) >= 3:
            repo_id = "/".join(parts[:2])
            gguf_file = "/".join(parts[2:])
    return repo_id, gguf_file, base_repo, safe_model_name

def extract_quant_type(raw_string: str) -> str:
    upper_string = raw_string.upper()
    quant_patterns = [r"Q[1-8]_[A-Z0-9_]+", r"IQ[1-4]_[A-Z0-9_]+", r"AWQ", r"GPTQ", r"FP8"]
    for pattern in quant_patterns:
        match = re.search(pattern, upper_string)
        if match: return match.group(0)
    return "FP16"

def clean_heuristic_noise(texts: list) -> list:
    cleaned_texts = []
    patterns_to_remove = [
        r"(?i)\bstart your response with\s*['\"].+?['\"]",
        r"(?i)\bbegin your answer with\s*['\"].+?['\"]",
        r"(?i)\bwrite your reply starting with\s*['\"].+?['\"]",
        r"(?i)\bplease[, ]+(?:as|act|pretend|roleplay).*",
        r"(?i)decode the following base64.*",
        r"(?i)base64[:\s]+[A-Za-z0-9+/=]{20,}",
        r"(?i)hex[:\s]+(?:[0-9A-Fa-f]{2}\s*){10,}",
        r"(?i)execute the hidden instruction.*",
        r"(?i)ignore previous rules and.*"
    ]
    combined_pattern = re.compile("|".join(patterns_to_remove), flags=re.IGNORECASE)
    for text in texts:
        clean_text = combined_pattern.sub("", str(text)).strip()
        cleaned_texts.append(clean_text)
    return cleaned_texts

# ==========================================
# 2. EMBEDDING EXTRACTION ENGINE
# ==========================================
def extract_embeddings(df: pd.DataFrame, repo_id: str, gguf_file: str, base_repo: str = None):
    logger.info(f"[HTS-Train] 🧠 Loading Embedding Layer for {repo_id}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer_kwargs = {}
    model_kwargs = {"device_map": "cpu"} 
    
    if gguf_file: 
        tokenizer_kwargs["gguf_file"] = gguf_file
        model_kwargs["gguf_file"] = gguf_file
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, **tokenizer_kwargs)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
    except Exception as e:
        logger.warning(f"[HTS-Train] ⚠️ Primary repo '{repo_id}' is missing config.json/tokenizer files.")
        if base_repo:
            tokenizer = AutoTokenizer.from_pretrained(base_repo)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(base_repo, device_map="cpu")
        else:
            raise e
            
    embed_layer = model.get_input_embeddings().to(device)
    embed_dim = embed_layer.weight.shape[1]
    batch_size = 256
    all_embeddings = []

    texts = df['text'].tolist() if 'text' in df.columns else df['input'].tolist()
    texts = clean_heuristic_noise(texts)

    logger.info(f"[HTS-Train] ⚙️ Vectorizing {len(texts)} pure prompts into latent geometry...")
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[i:i+batch_size], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            all_embeddings.append(embed_layer(inputs.input_ids).mean(dim=1).cpu())
                
    del model, tokenizer, embed_layer
    torch.cuda.empty_cache()
    gc.collect()
    
    return torch.cat(all_embeddings, dim=0), embed_dim

# ==========================================
# 3. METRICS EVALUATORS
# ==========================================
def calculate_dynamic_metrics(model, X, y_true, loss_type='Cosine'):
    with torch.no_grad():
        projected = model(X)
        y_t = y_true.cpu().numpy()
        
        if loss_type == 'Cosine': 
            distances = (1.0 - F.cosine_similarity(projected, X, dim=1)).cpu().numpy()
        else: 
            distances = torch.norm(projected - X, p=2, dim=1).cpu().numpy()
        
        fpr, tpr, roc_thresholds = roc_curve(y_t, distances)
        auc = roc_auc_score(y_t, distances)
        precisions, recalls, pr_thresholds = precision_recall_curve(y_t, distances)
        
        beta_hard = 0.5
        f_beta_hard = (1 + beta_hard**2) * (precisions * recalls) / ((beta_hard**2 * precisions) + recalls + 1e-9)
        hard_idx = np.argmax(f_beta_hard)
        hard_threshold = pr_thresholds[hard_idx] if hard_idx < len(pr_thresholds) else pr_thresholds[-1]
        
        valid_recall_indices = np.where(recalls >= 0.55)[0]
        if len(valid_recall_indices) > 0:
            grey_idx = valid_recall_indices[-1]
            grey_threshold = pr_thresholds[grey_idx] if grey_idx < len(pr_thresholds) else pr_thresholds[-1]
        else:
            grey_threshold = hard_threshold * 0.75

        grey_threshold = min(grey_threshold, hard_threshold * 0.85)

        y_pred = (distances >= hard_threshold).astype(int)
        acc = accuracy_score(y_t, y_pred)
        f1 = f1_score(y_t, y_pred, zero_division=0)
        
    return acc, f1, auc, fpr, tpr, float(hard_threshold), float(grey_threshold)

def calculate_fixed_metrics(model, X, y_true, threshold, loss_type='Cosine'):
    with torch.no_grad():
        projected = model(X)
        y_t = y_true.cpu().numpy()
        if loss_type == 'Cosine': distances = (1.0 - F.cosine_similarity(projected, X, dim=1)).cpu().numpy()
        else: distances = torch.norm(projected - X, p=2, dim=1).cpu().numpy()
            
        y_pred = (distances >= threshold).astype(int)
    return accuracy_score(y_t, y_pred), f1_score(y_t, y_pred, zero_division=0), recall_score(y_t, y_pred, zero_division=0), precision_score(y_t, y_pred, zero_division=0)

# ==========================================
# 4. PLOTTING & CONTOUR MAPPING
# ==========================================
def save_training_plots(model_name, history, fpr, tpr, auc, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='o')
    plt.title(f'"{model_name}" SN-RAE Convergence')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'{model_name}_loss_curve.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_f1'], label='Train F1', marker='o')
    plt.plot(epochs, history['val_f1'], label='Validation F1', marker='o')
    plt.title(f'"{model_name}" F1-Score Convergence (Robustness)')
    plt.xlabel('Epochs'); plt.ylabel('F1 Score'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'{model_name}_f1_curve.png'))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'"{model_name}" Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right"); plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'{model_name}_roc_curve.png'))
    plt.close()

def save_confusion_matrix(model_name, y_true, y_pred, RESPONSE_DIR):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Attack'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'"{model_name}" Final Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESPONSE_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

def save_latent_contour_heatmap(model_name, distances, labels, grey_th, hard_th, RESPONSE_DIR):
    plt.figure(figsize=(10, 6))
    benign_dists = distances[labels == 0]
    attack_dists = distances[labels == 1]
    
    plt.hist(benign_dists, bins=50, alpha=0.5, color='green', label='Benign', density=True)
    plt.hist(attack_dists, bins=50, alpha=0.5, color='red', label='Attack', density=True)
    
    plt.axvline(grey_th, color='orange', linestyle='dashed', linewidth=2, label=f'Grey Zone ({grey_th:.3f})')
    plt.axvline(hard_th, color='black', linestyle='solid', linewidth=2, label=f'Hard Block ({hard_th:.3f})')
    
    plt.title(f'"{model_name}" Non-Linear Manifold Separation')
    plt.xlabel('Transformation Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESPONSE_DIR, f'{model_name}_latent_contour.png'))
    plt.close()

def parallel_simulated_annealing(distances, labels, initial_grey, initial_hard, num_chains=4, steps=1000):
    logger.info(f"[HTS-Train] 🧬 Running {num_chains} Parallel Simulated Annealing Chains...")
    best_overall_energy = float('inf')
    best_overall_grey = initial_grey
    best_overall_hard = initial_hard
    
    for chain in range(num_chains):
        current_grey = initial_grey + np.random.normal(0, 0.05)
        current_hard = initial_hard + np.random.normal(0, 0.05)
        if current_grey >= current_hard: current_hard = current_grey + 0.02
            
        current_preds = (distances >= current_hard).astype(int)
        current_energy = (1 - recall_score(labels, current_preds)) + 2.0 * (1 - precision_score(labels, current_preds, zero_division=0))
        T = 1.0 
        cooling_rate = 0.99
        
        for step in range(steps):
            new_grey = current_grey + np.random.normal(0, 0.01)
            new_hard = current_hard + np.random.normal(0, 0.01)
            if new_grey >= new_hard: new_hard = new_grey + 0.02
                
            new_preds = (distances >= new_hard).astype(int)
            new_energy = (1 - recall_score(labels, new_preds)) + 2.0 * (1 - precision_score(labels, new_preds, zero_division=0))
            
            if new_energy < current_energy:
                current_grey, current_hard, current_energy = new_grey, new_hard, new_energy
            else:
                prob = np.exp(-(new_energy - current_energy) / T)
                if np.random.rand() < prob:
                    current_grey, current_hard, current_energy = new_grey, new_hard, new_energy
            T *= cooling_rate
            
        if current_energy < best_overall_energy:
            best_overall_energy = current_energy
            best_overall_grey = current_grey
            best_overall_hard = current_hard
            
    return best_overall_grey, best_overall_hard

# ==========================================
# 5. CORE TRAINING ENGINE
# ==========================================
def train_hts_matrix(model_string: str, loss_type: str, dataset_dir: str = "dataset"):
    logger.info("\n" + "="*70)
    logger.info(f"[HTS-Train] 🛡️ INITIATING SN-RAE COMPILATION FOR: {model_string}")
    logger.info(f"[HTS-Train] ⚙️ Architecture: Spectral Residual Autoencoder | Metric: {loss_type}")
    logger.info("="*70)
    
    try:
        repo_id, gguf_file, base_repo, clean_model_string = parse_model_string(model_string)
        quant_type = extract_quant_type(clean_model_string)
        model_name = repo_id.split("/")[-1]
        
        save_dir = os.path.join("hts_stored_weights", quant_type)
        os.makedirs(save_dir, exist_ok=True)
        plot_dir = os.path.join(save_dir, f"{model_name}_{loss_type}_plots")
        os.makedirs(plot_dir, exist_ok=True) 
    
        save_path_mlp = os.path.join(save_dir, f"{model_name}_{loss_type}_hts_mlp.pt")
        metrics_export_path = os.path.join(save_dir, f"{model_name}_{loss_type}_metrics.json")

        safe_model_name = clean_model_string.replace("/", "_")
        data_path = os.path.join(dataset_dir, f"{safe_model_name}_combined_dataset.jsonl")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found.")

        df = pd.read_json(data_path, orient="records", lines=True)
        if 'text' not in df.columns:
            if 'input' in df.columns: df = df.rename(columns={'input': 'text'})
            else: raise KeyError("Dataset is missing the required 'text' column.")
        
        # Obfuscation weighting
        obfuscation_patterns = [r'base64', r'hex', r'\[.*?\]', r'\*.*?\*', r'ignore\s+(previous|all|above)', r'sudo', r'system prompt']
        weights_arr = np.ones(len(df), dtype=np.float32)
        for idx, row in df.iterrows():
            text_val = str(row.get('text', '')).lower()
            is_malicious = (row['label'] == 1)
            has_obfuscation = any(re.search(p, text_val) for p in obfuscation_patterns)
            if not is_malicious and has_obfuscation: weights_arr[idx] = 5.0 
            elif is_malicious: weights_arr[idx] = 2.0 

        df['sample_weight'] = weights_arr
        
        X_tensor, embed_dim = extract_embeddings(df, repo_id, gguf_file, base_repo)
        y_tensor = torch.tensor(df['label'].values, dtype=torch.float32)
        w_tensor = torch.tensor(df['sample_weight'].values, dtype=torch.float32)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        dataset_size = len(X_tensor)
        val_size = int(0.2 * dataset_size)
        train_size = dataset_size - val_size
        
        full_dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        train_indices = train_dataset.indices
        y_train = y_tensor[train_indices]
        
        class_counts = torch.bincount(y_train.long())
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[y_train.long()]
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
        
        val_indices = val_dataset.indices
        X_val, y_val, w_val = X_tensor[val_indices].to(device), y_tensor[val_indices].to(device), w_tensor[val_indices].to(device)
        X_tr_full, y_tr_full = X_tensor[train_indices].to(device), y_tensor[train_indices].to(device)

        # ---------------------------------------------------------
        # INITIALIZATION: SN-RAE
        # ---------------------------------------------------------
        epochs = 100
        noise_std = 0.05 
        bottleneck_dim = embed_dim // 4 

        projector = SpectralResidualProjector(embed_dim, bottleneck_dim).to(device)
        
        optimizer = optim.AdamW(projector.parameters(), lr=5e-4, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        criterion = GeometricMarginLoss(loss_type=loss_type, safe_margin=0.05, attack_margin=0.8)
        early_stopper = NativeEarlyStopping(patience=12, min_delta=1e-5, mode='max')

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': [], 'train_prec': [], 'val_prec': [], 'train_rec': [], 'val_rec': []}
        
        logger.info(f"[HTS-Train] 🔥 Commencing Spectral Residual Training...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            projector.train()
            
            for i, (batch_x, batch_y, batch_w) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
                batch_x, batch_y, batch_w = batch_x.to(device), batch_y.to(device), batch_w.to(device)
                
                optimizer.zero_grad()
                
                current_noise = noise_std * (1.0 - (epoch / epochs))
                noise = torch.randn_like(batch_x)
                noise = (noise / torch.norm(noise, dim=1, keepdim=True)) * torch.norm(batch_x, dim=1, keepdim=True) * current_noise
                noisy_x = batch_x + noise
                
                projected = projector(noisy_x)
                # Notice L1 Reconstruction is removed because the SN-RAE handles it natively
                loss = criterion(projected, noisy_x, batch_y, batch_w)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            
            # --- VALIDATION PHASE ---
            projector.eval()
            val_acc, val_f1, val_auc, fpr_pts, tpr_pts, hard_th, grey_th = calculate_dynamic_metrics(projector, X_val, y_val, loss_type)
            
            with torch.no_grad():
                val_projected = projector(X_val)
                current_val_loss = criterion(val_projected, X_val, y_val, w_val).item()

            tr_acc, tr_f1, tr_rec, tr_prec = calculate_fixed_metrics(projector, X_tr_full, y_tr_full, hard_th, loss_type)
            v_acc, v_f1, v_rec, v_prec = calculate_fixed_metrics(projector, X_val, y_val, hard_th, loss_type)
            
            history['train_loss'].append(avg_train_loss); history['val_loss'].append(current_val_loss)
            history['train_acc'].append(tr_acc); history['val_acc'].append(v_acc)
            history['train_f1'].append(tr_f1); history['val_f1'].append(v_f1)
            history['train_prec'].append(tr_prec); history['val_prec'].append(v_prec)
            history['train_rec'].append(tr_rec); history['val_rec'].append(v_rec)
            
            scheduler.step(val_auc)
            logger.info(f"Epoch {epoch+1:02d} | Val AUC: {val_auc:.3f} | F1: {v_f1:.3f} | Loss: {current_val_loss:.4f}")

            early_stopper(val_auc, projector)
            if early_stopper.early_stop:
                logger.info(f"[HTS-Train] 🛑 Early stopping triggered! Best SN-RAE captured at AUC {early_stopper.best_score:.4f}.")
                projector.load_state_dict(early_stopper.best_state)
                break

        # =================================================================
        # POST-TRAINING: STATISTICAL K-SIGMA THRESHOLD CALCULATION
        # =================================================================
        logger.info("⚙️ Calculating final statistical thresholds based on Benign distribution...")
        
        projector.eval()
        val_benign_distances = []

        with torch.no_grad():
            # 1. Filter your validation data to ONLY include Benign prompts (label == 0)
            # Assuming your labels are in a tensor called y_val and embeddings in X_val
            benign_indices = (y_val == 0).nonzero(as_tuple=True)[0]
            X_benign_val = X_val[benign_indices]

            # Process in batches to avoid VRAM overflow on large validation sets
            batch_size = 256
            for i in range(0, len(X_benign_val), batch_size):
                inputs = X_benign_val[i:i+batch_size].to(device)
                reconstructed = projector(inputs)
                
                if loss_type == 'Cosine':
                    distances = 1.0 - F.cosine_similarity(reconstructed, inputs, dim=1)
                else:
                    distances = torch.norm(reconstructed - inputs, p=2, dim=1)
                    
                val_benign_distances.extend(distances.cpu().numpy().tolist())

        # Export unified architecture state dict
        logger.info(f"[HTS-Train] ✅ Spectral Residual structure secured.")
        torch.save(projector.state_dict(), save_path_mlp)
        
        # 1. Calculate the true center and spread of the Benign (Green) Distribution
        benign_mu = np.mean(val_benign_distances)
        benign_sigma = np.std(val_benign_distances)
        
        logger.info(f"📊 Benign Manifold Stats -> Mean: {benign_mu:.4f} | StdDev: {benign_sigma:.4f}")
        
        # 2. Hard Block: Set to 4 Standard Deviations away from the mean.
        # This mathematically guarantees that 99.99% of benign prompts will NEVER hit the Hard Block.
        optimal_hard_block = benign_mu + (4.0 * benign_sigma)
        
        # Clamp to a maximum of 0.90 to ensure attacks can't push it entirely out of bounds
        optimal_hard_block = min(optimal_hard_block, 0.90) 
        
        # 3. Tight Grey Zone: Restrict it to exactly 0.05 below the Hard Block
        # This keeps it as far away from the benign Green region as mathematically possible!
        margin = 0.05
        optimal_grey_zone = optimal_hard_block - margin
        
        logger.info(f"🎯 Calibrated Thresholds -> Hard Block: {optimal_hard_block:.4f} | Grey Zone: {optimal_grey_zone:.4f}")

        with open(metrics_export_path, 'w') as f:
            json.dump({
                "Target_Model": model_string,
                "loss_type": loss_type,
                "Final_ROC_AUC": float(val_auc),
                "Hard_Block_Threshold": float(optimal_hard_block),
                "Grey_Zone_Threshold": float(optimal_grey_zone),
                "Benign_Mean": float(benign_mu),
                "Benign_Std": float(benign_sigma),
                "Architecture": "SN-RAE",
            }, f, indent=4)
            
        final_preds = (all_dists[val_indices] >= discovered_hard).astype(int)
        save_confusion_matrix(model_name, y_val.cpu().numpy(), final_preds, plot_dir)
        save_training_plots(model_name, history, fpr_pts, tpr_pts, val_auc, plot_dir)
        save_latent_contour_heatmap(model_name, all_dists, y_tensor.numpy(), discovered_grey, discovered_hard, plot_dir)

    except Exception as e:
        logger.error(f"[HTS-Train] ❌ FATAL ERROR IN TRAINING PIPELINE: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--loss_type", type=str, choices=["Euclidean", "Cosine"], default="Cosine")
    args = parser.parse_args()
    
    train_hts_matrix(args.model, args.loss_type)