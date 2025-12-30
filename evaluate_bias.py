import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

CSV_PATH = "data/dataset.csv"
PLOT_DIR = "data/plots"

def run_bias_check():
    if not os.path.exists(CSV_PATH):
        print("❌ dataset.csv not found. Run crawler.py first.")
        return
    
    if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

    print("⚖️ Running Protocol E: Dataset Bias Evaluation...")
    df = pd.read_csv(CSV_PATH)
    
    df['hemisphere'] = df['latitude'].apply(lambda x: 'North' if x > 0 else 'South')
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x='hemisphere', data=df, palette='coolwarm')
    plt.title("Figure 3.4: Hemisphere Distribution (Bias Check)")
    plt.savefig(f"{PLOT_DIR}/bias_hemisphere.png")
    
    plt.figure(figsize=(10, 6))
    plt.hexbin(df['longitude'], df['latitude'], gridsize=20, cmap='inferno', mincnt=1)
    plt.colorbar(label='Image Count')
    plt.title("Figure 3.5: Geographic Density (Sampling Bias)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(f"{PLOT_DIR}/bias_geographic_density.png")
    
    if 'provider' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(y='provider', data=df, palette='viridis')
        plt.title("Figure 3.6: Data Source Distribution")
        plt.savefig(f"{PLOT_DIR}/bias_providers.png")

    print("✅ Bias Report Generated in data/plots/")

if __name__ == "__main__":
    run_bias_check()