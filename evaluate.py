import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def evaluate_model(model, eval_dataset, config, device):
    """Evaluate model on validation set"""

    model.eval()
    predictions = []
    probabilities = []
    labels =[]

    #Create dataloader
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size = config.per_device_eval_batch_size,
        shuffle=False
    )

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch)

            #Get Predictions and probabilities
            probs = torch.softmax(outputs.logits, dim =-1)
            preds = torch.argmax(outputs.logits, dim =-1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs[:,1].cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())


        #Calculate Metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='binary'),
             'recall': recall_score(labels, predictions, average='binary'),
            'f1_score': f1_score(labels, predictions, average='binary'),
            'roc_auc': roc_auc_score(labels, probabilities)
            }

            # Calcule ROC curve data
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
            }

    return metrics

def visualize_metrics(metrics_history, output_dir):
    """Create comprehensive visualizations for metrics"""
    os.makedirs(output_dir, exist_ok=True)

    #Set Style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    #Metrics over time
    if len(metrics_history) > 1:
        fig, axes = plt.subplots(2,3, figsize = (15, 10))
        axes = axes.flatten()

        epochs = range(1,len(metrics_history) + 1)
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

        for idx, metric in enumerate(metric_names):
            values = [m[metric] for m in metrics_history]
            axes[idx].plot(epochs, values, marker ='o', linewidth=2, markersize=8)
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Over Epochs', fontsize=14)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.replace("_", " ").title())
            axes[idx].set_ylim([0, 1.05])
            axes[idx].grid(True, alpha=0.3)


            # Add value labels
            for x, y in zip(epochs, values):
                axes[idx].text(x,y + 0.02, f'{y:.3f}', ha='center', fontsize=10)

        
        #Remove extra subplot
        fig.delaxes(axes[5])

        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()

        #Bar Chart
        latest_metrics = metrics_history[-1] if metrics_history else {}
        if latest_metrics:
            fig, ax = plt.subplots(figsize=(10,6))

            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            values = [latest_metrics[m] for m in metrics]
            labels = [m.replace('_', ' ').title() for m in metrics]
            bars = ax.bar(labels, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height+ 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            ax.set_ylim([0,1.1])
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Model Performance Metrics', fontsize=16, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)

            # Add horizontal line at 0.9 for reference
            ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
            ax.legend()

            plt.tight_layout()
            plt.savefig(f'{output_dir}/final_metrics_bar.png', dpi=300, bbox_inches='tight')
            plt.close()

        
        # ROC Curve
    if 'roc_curve' in latest_metrics:
            fig, ax = plt.subplots(figsize=(8,8))

            fpr = latest_metrics['roc_curve']['fpr']
            tpr = latest_metrics['roc_curve']['tpr']
            auc = latest_metrics['roc_auc']

            ax.plot(fpr, tpr, color='#FF6B6B', linewidth = 3,
                    label=f'ROC Curve (AUC = {auc:.3f})')
            ax.plot([0,1], [0,1], 'k--', linewidth=2, label= 'Random Classifier')

            ax.fill_between(fpr, tpr, alpha =0.3, color='#FF6B6B')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
            ax.legend(loc='lower right', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

    if latest_metrics:
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
            values = [latest_metrics['accuracy'], latest_metrics['precision'], 
                    latest_metrics['recall'], latest_metrics['f1_score'], 
                    latest_metrics['roc_auc']]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='#4ECDC4')
            ax.fill(angles, values, alpha=0.25, color='#4ECDC4')
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
            ax.grid(True)
            
            ax.set_title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/metrics_radar.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def create_metrics_comparison_plot(baseline_metrics, lora_metrics, output_path):
     """Compare basline Vs LoRA metrics"""

     fig, ax = plt.subplots(figsize=(12,6))

     metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
     x = np.arange(len(metrics))
     width = 0.35
     baseline_values = [baseline_metrics.get(m.lower().replace(' ', '_'), 0) for m in metrics]
     lora_values = [lora_metrics.get(m.lower().replace(' ', '_'), 0) for m in metrics]
    
     bars1 = ax.bar(x - width/2, baseline_values, width, label='Full Fine-tuning', color='#3498db')
     bars2 = ax.bar(x + width/2, lora_values, width, label='LoRA', color='#e74c3c')
    
     # Add value labels
     for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
     ax.set_ylabel('Score', fontsize=12)
     ax.set_title('Full Fine-tuning vs LoRA Performance', fontsize=16, fontweight='bold')
     ax.set_xticks(x)
     ax.set_xticklabels(metrics)
     ax.legend()
     ax.set_ylim(0, 1.15)
     ax.grid(True, axis='y', alpha=0.3)
    
     plt.tight_layout()
     plt.savefig(output_path, dpi=300, bbox_inches='tight')
     plt.close()

def save_metrics(metrics, filepath):
     """Save metrics to JSON file"""

     metrics_to_save = {k: v for k, v in metrics.items() if k != 'roc_curve'}
     with open(filepath, 'w') as f:
          json.dump(metrics_to_save, f, indent = 2)
    
    


    

        