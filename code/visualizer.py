# ./visualizer.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import traceback
import argparse

from logger_config import logger

def generate_academic_heatmaps(csv_path, chart_type="asr_heatmap"):
    logger.info("\n" + "="*70)
    logger.info(f"[VISUALIZER] 🟢 Entering TRY block: Generating {chart_type.upper()}")
    
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        if not os.path.exists(csv_path):
            logger.warning(f"[VISUALIZER] ⚠️ Data not found at {csv_path}. Creating an empty placeholder.")
            pd.DataFrame().to_csv(csv_path, index=False)
            return  # Gracefully exit without crashing

        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()

        if df.empty:
            logger.warning("[VISUALIZER] ⚠️ DataFrame is empty. No charts to plot.")
            return

        logger.info(f"[VISUALIZER] 📊 Data loaded successfully from {csv_path}. Generating Academic Matrix...")
        sns.set_theme(style="white")

        # =========================================================
        # 🟥 ASR HEATMAP (RED) - For Global Attack Results
        # =========================================================
        if chart_type == "asr_heatmap":
            summary_df = df.groupby(['Defense_Strategy', 'Attack', 'Model'])['Jailbroken'].mean().reset_index()
            summary_df['ASR (%)'] = summary_df['Jailbroken'] * 100

            defenses = summary_df['Defense_Strategy'].unique()
            num_defenses = len(defenses)
            
            logger.info(f"[VISUALIZER] ⚙️ Found {num_defenses} defense strategies to plot: {list(defenses)}")

            cols = 2
            rows = int(np.ceil(num_defenses / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(16 * cols, 8 * rows))
            if rows == 1 and cols == 1: axes = np.array([axes])
            axes = axes.flatten()

            for i, strategy in enumerate(defenses):
                ax = axes[i]
                data = summary_df[summary_df['Defense_Strategy'] == strategy]
                
                # Pivot the data for the Heatmap: Rows = Models, Cols = Attacks
                pivot_data = data.pivot(index='Model', columns='Attack', values='ASR (%)')
                
                sns.heatmap(
                    pivot_data, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=100, ax=ax,
                    linewidths=1, linecolor='black', annot_kws={"size": 14, "weight": "bold"},
                    cbar_kws={'label': 'Attack Success Rate (%)'} if i % cols == cols - 1 else None
                )
                
                ax.set_title(f"Defense Strategy: {strategy.upper()}", fontsize=18, fontweight='bold', pad=15)
                ax.set_ylabel("Target Model", fontsize=14, fontweight='bold')
                ax.set_xlabel("Adversarial Attack Method", fontsize=14, fontweight='bold')
                ax.tick_params(axis='y', rotation=0, labelsize=12)
                ax.tick_params(axis='x', rotation=45, labelsize=12)

            for j in range(num_defenses, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle("Attack Success Rate (ASR) Matrix by Defense Strategy", fontsize=24, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            os.makedirs("./results", exist_ok=True)
            output_path = "./results/ACADEMIC_ASR_HEATMAP_MATRIX.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[VISUALIZER] ✅ Master ASR Heatmap saved for publication -> {output_path}")

        # =========================================================
        # 🟩 FPR HEATMAP (GREEN) - For Global Benign Results
        # =========================================================
        elif chart_type == "fpr_heatmap":
            # Ensure boolean conversion for Blocked_By_Defense
            df['Blocked_By_Defense'] = df['Blocked_By_Defense'].astype(bool)
            
            # Aggregate False Positive Rate across Models and Defenses
            summary_df = df.groupby(['Defense_Strategy', 'Model'])['Blocked_By_Defense'].mean().reset_index()
            summary_df['FPR (%)'] = summary_df['Blocked_By_Defense'] * 100
            
            # Pivot: Rows = Target Models, Columns = Defense Strategies
            pivot_data = summary_df.pivot(index='Model', columns='Defense_Strategy', values='FPR (%)')
            
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            
            sns.heatmap(
                pivot_data, annot=True, fmt=".1f", cmap="Greens", vmin=0, vmax=100, ax=ax,
                linewidths=1, linecolor='black', annot_kws={"size": 14, "weight": "bold"},
                cbar_kws={'label': 'False Positive Rate (%)'}
            )
            
            ax.set_title("False Positive Rate (FPR) Matrix on Benign Prompts\n(Lower is Better)", fontsize=20, fontweight='bold', pad=20)
            ax.set_ylabel("Target Model", fontsize=14, fontweight='bold')
            ax.set_xlabel("Defense Strategy", fontsize=14, fontweight='bold')
            ax.tick_params(axis='y', rotation=0, labelsize=12)
            ax.tick_params(axis='x', rotation=0, labelsize=12)
            
            plt.tight_layout()
            
            os.makedirs("./results", exist_ok=True)
            output_path = "./results/ACADEMIC_FPR_HEATMAP_MATRIX.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[VISUALIZER] ✅ Master FPR Heatmap saved for publication -> {output_path}")

    except Exception as e:
        logger.error(f"[VISUALIZER] ❌ EXCEPTION during Master Chart Generation: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info("[VISUALIZER] 🏁 Exiting FINALLY block")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Academic Matrix Heatmaps")
    parser.add_argument("--input", type=str, required=True, help="Path to the CSV results file")
    parser.add_argument("--type", type=str, choices=["asr_heatmap", "fpr_heatmap"], required=True, help="Type of heatmap to generate")
    
    args = parser.parse_args()
    generate_academic_heatmaps(csv_path=args.input, chart_type=args.type)