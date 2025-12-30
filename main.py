import os
import sys

def run_command(script_name):
    print(f"\n{'='*60}")
    print(f"🎬 RUNNING: {script_name}")
    print(f"{'='*60}")
    exit_code = os.system(f"python {script_name}")
    if exit_code != 0:
        print(f"❌ Error running {script_name}")

def main():
    # 1. Data Acquisition
    run_command("crawler.py") 
    run_command("cleaner.py") 
    run_command("cleaner_sorter.py") 
    
    # 2. Baselines (Context)
    run_command("baseline_ml.py")   # Classical ML (Random Forest)
    run_command("train.py")         # Deep Learning Baseline (Custom CNN)
    
    run_command("train_transfer.py") # Transfer Learning (MobileNetV2)
    
    # 4. Evaluation Suite
    run_command("train_kfold.py")
    run_command("visualize_results.py") 
    run_command("test_robustness.py") 
    run_command("advanced_analytics.py")
    run_command("evaluate_bias.py")     
    run_command("extra_analytics.py")     

    
    # 5. Launch App
    print("\n✅ Pipeline Complete. Launching Dashboard...")
    os.system("streamlit run app.py")

if __name__ == "__main__":
    main()