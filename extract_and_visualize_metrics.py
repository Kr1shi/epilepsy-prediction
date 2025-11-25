import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Directory containing result text files
result_dir = "/home/krishi/Projects/epilepsy-prediction/result_text"

# Dictionary to store metrics for each subject
metrics_data = {
    'subjects': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'auc_roc': []
}

# Pattern to extract metrics after BATCH EVALUATION SUMMARY
# Looking for the first occurrence of each metric value after the summary header
pattern = r'BATCH EVALUATION SUMMARY\s+BATCH EVALUATION SUMMARY\s+={60}\s+={60}\s+Folds evaluated:.*?Mean metrics \(across folds\):\s+Mean metrics \(across folds\):\s+Accuracy:\s+([\d.]+)\s+\(±[\d.]+\)\s+Accuracy:\s+([\d.]+).*?Precision:\s+([\d.]+)\s+\(±[\d.]+\)\s+Precision:\s+([\d.]+).*?Recall:\s+([\d.]+)\s+\(±[\d.]+\)\s+Recall:\s+([\d.]+).*?F1 Score:\s+([\d.]+)\s+\(±[\d.]+\)\s+F1 Score:\s+([\d.]+).*?AUC-ROC:\s+([\d.]+)\s+\(±[\d.]+\)\s+AUC-ROC:\s+([\d.]+)'

# Get all result files sorted numerically
result_files = sorted(os.listdir(result_dir), key=lambda x: int(x) if x.isdigit() else float('inf'))

print("Extracting metrics from BATCH EVALUATION SUMMARY sections...")
print("-" * 70)

for filename in result_files:
    if not filename.isdigit():
        continue

    filepath = os.path.join(result_dir, filename)

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Search for the metrics pattern
        match = re.search(pattern, content, re.DOTALL)

        if match:
            # Since metrics are duplicated, just take the first one
            accuracy = float(match.group(1))
            precision = float(match.group(3))
            recall = float(match.group(5))
            f1_score = float(match.group(7))
            auc_roc = float(match.group(9))

            subject_id = f"CHB{filename.zfill(2)}"

            metrics_data['subjects'].append(subject_id)
            metrics_data['accuracy'].append(accuracy)
            metrics_data['precision'].append(precision)
            metrics_data['recall'].append(recall)
            metrics_data['f1_score'].append(f1_score)
            metrics_data['auc_roc'].append(auc_roc)

            print(f"{subject_id}: Acc={accuracy:.4f}, Prec={precision:.4f}, "
                  f"Rec={recall:.4f}, F1={f1_score:.4f}, AUC={auc_roc:.4f}")
        else:
            print(f"⚠️  {filename}: Metrics not found in BATCH EVALUATION SUMMARY")

    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")

print("-" * 70)
print(f"Successfully extracted metrics for {len(metrics_data['subjects'])} subjects\n")

# Create the bar chart
if metrics_data['subjects']:
    plt.figure(figsize=(14, 8))

    x = np.arange(len(metrics_data['subjects'])) * 1.5  # Increase spacing between patient groups
    width = 0.15  # Width of each bar

    # Plot each metric as grouped bars
    plt.bar(x - 2*width, metrics_data['accuracy'], width,
            label='Accuracy', color='#1f77b4')
    plt.bar(x - width, metrics_data['precision'], width,
            label='Precision', color='#ff7f0e')
    plt.bar(x, metrics_data['recall'], width,
            label='Recall', color='#2ca02c')
    plt.bar(x + width, metrics_data['f1_score'], width,
            label='F1 Score', color='#d62728')
    plt.bar(x + 2*width, metrics_data['auc_roc'], width,
            label='AUC-ROC', color='#9467bd')

    # Customize the plot
    plt.xlabel('Subject', fontsize=13, fontweight='bold')
    plt.ylabel('Score', fontsize=13, fontweight='bold')
    plt.title('Seizure Prediction Model Performance Metrics Across Subjects',
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, metrics_data['subjects'], rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim([0, 1.05])  # Full range from 0 to 1
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='lower right', framealpha=0.95)
    plt.tight_layout()

    # Save the plot
    output_path = "/home/krishi/Projects/epilepsy-prediction/metrics_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_path}")

    # Also create a summary statistics table
    print("\nSummary Statistics:")
    print("-" * 70)
    print(f"{'Metric':<15} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
    print("-" * 70)

    for metric_name, metric_values in [
        ('Accuracy', metrics_data['accuracy']),
        ('Precision', metrics_data['precision']),
        ('Recall', metrics_data['recall']),
        ('F1 Score', metrics_data['f1_score']),
        ('AUC-ROC', metrics_data['auc_roc'])
    ]:
        mean_val = np.mean(metric_values)
        std_val = np.std(metric_values)
        min_val = np.min(metric_values)
        max_val = np.max(metric_values)
        print(f"{metric_name:<15} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")

else:
    print("❌ No metrics found in any result files.")
