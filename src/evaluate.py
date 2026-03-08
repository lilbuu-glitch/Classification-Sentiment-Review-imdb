import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve, auc)
import os

from .utils import setup_logging, ensure_dir, save_metrics, create_confusion_matrix_plot, create_roc_curve_plot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and visualization."""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_model(self, 
                      model, 
                      X_test, 
                      y_test, 
                      model_name: str = "model",
                      label_encoder=None,
                      save_plots: bool = True,
                      plot_dir: str = "plots") -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for reporting
            label_encoder: Label encoder for class names
            save_plots: Whether to save evaluation plots
            plot_dir: Directory to save plots
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {model_name} on test set...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            # For SVM, use decision function
            y_proba = model.decision_function(X_test)
            # Normalize to [0, 1] range
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(y_test, y_pred, y_proba)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Get class names if label encoder is provided
        class_names = None
        if label_encoder is not None:
            class_names = label_encoder.classes_.tolist()
        
        # Create plots
        if save_plots:
            ensure_dir(plot_dir)
            
            # Confusion matrix
            cm_path = os.path.join(plot_dir, f"{model_name}_confusion_matrix.png")
            create_confusion_matrix_plot(y_test, y_pred, labels=class_names, save_path=cm_path)
            
            # ROC curve
            if y_proba is not None:
                roc_path = os.path.join(plot_dir, f"{model_name}_roc_curve.png")
                create_roc_curve_plot(y_test, y_proba, save_path=roc_path)
        
        # Store results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': class_report,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist() if y_proba is not None else None,
            'true_labels': y_test.tolist(),
            'class_names': class_names
        }
        
        self.evaluation_results[model_name] = results
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1-score: {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] > 0:
            logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return results
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_proba=None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_micro': precision_score(y_true, y_pred, average='micro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Add ROC-AUC if probabilities are available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        # Calculate per-class metrics for binary classification
        unique_labels = np.unique(y_true)
        if len(unique_labels) == 2:
            # Binary classification metrics
            metrics['precision_binary'] = precision_score(y_true, y_pred, pos_label=1)
            metrics['recall_binary'] = recall_score(y_true, y_pred, pos_label=1)
            metrics['f1_binary'] = f1_score(y_true, y_pred, pos_label=1)
        
        return metrics
    
    def compare_models(self, model_names: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple models' performance.
        
        Args:
            model_names: List of model names to compare. If None, compare all evaluated models.
            
        Returns:
            DataFrame with model comparison
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name not in self.evaluation_results:
                logger.warning(f"Model {model_name} not found in evaluation results")
                continue
            
            results = self.evaluation_results[model_name]
            metrics = results['metrics']
            
            row = {
                'model': model_name,
                'accuracy': metrics['accuracy'],
                'f1_weighted': metrics['f1_weighted'],
                'precision_weighted': metrics['precision_weighted'],
                'recall_weighted': metrics['recall_weighted']
            }
            
            if 'roc_auc' in metrics:
                row['roc_auc'] = metrics['roc_auc']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('f1_weighted', ascending=False)
        
        return comparison_df
    
    def generate_evaluation_report(self, model_name: str, save_path: str = None) -> str:
        """
        Generate a detailed evaluation report for a model.
        
        Args:
            model_name: Name of the model to report on
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        results = self.evaluation_results[model_name]
        metrics = results['metrics']
        class_report = results['classification_report']
        
        report = f"=== EVALUATION REPORT FOR {model_name.upper()} ===\n\n"
        
        # Overall metrics
        report += "OVERALL METRICS:\n"
        report += f"Accuracy: {metrics['accuracy']:.4f}\n"
        report += f"F1-score (weighted): {metrics['f1_weighted']:.4f}\n"
        report += f"Precision (weighted): {metrics['precision_weighted']:.4f}\n"
        report += f"Recall (weighted): {metrics['recall_weighted']:.4f}\n"
        
        if 'roc_auc' in metrics:
            report += f"ROC-AUC: {metrics['roc_auc']:.4f}\n"
        
        report += "\n"
        
        # Detailed classification report
        report += "DETAILED CLASSIFICATION REPORT:\n"
        report += classification_report(results['true_labels'], results['predictions'], 
                                     target_names=results['class_names'])
        
        # Confusion matrix analysis
        report += "\nCONFUSION MATRIX ANALYSIS:\n"
        cm = confusion_matrix(results['true_labels'], results['predictions'])
        report += f"True Positives: {cm[1,1] if cm.shape[0] > 1 else cm[0,0]}\n"
        report += f"True Negatives: {cm[0,0] if cm.shape[0] > 1 else 'N/A'}\n"
        report += f"False Positives: {cm[0,1] if cm.shape[0] > 1 else 'N/A'}\n"
        report += f"False Negatives: {cm[1,0] if cm.shape[0] > 1 else 'N/A'}\n"
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def save_all_results(self, save_dir: str = "evaluation_results"):
        """
        Save all evaluation results to files.
        
        Args:
            save_dir: Directory to save results
        """
        ensure_dir(save_dir)
        
        # Save individual model results
        for model_name, results in self.evaluation_results.items():
            model_dir = os.path.join(save_dir, model_name)
            ensure_dir(model_dir)
            
            # Save metrics
            metrics_path = os.path.join(model_dir, "metrics.json")
            save_metrics(results['metrics'], metrics_path)
            
            # Save classification report
            report_path = os.path.join(model_dir, "classification_report.json")
            save_metrics(results['classification_report'], report_path)
            
            # Save predictions
            predictions_df = pd.DataFrame({
                'true_labels': results['true_labels'],
                'predictions': results['predictions'],
                'probabilities': results['probabilities'] if results['probabilities'] else [None] * len(results['predictions'])
            })
            predictions_path = os.path.join(model_dir, "predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)
            
            # Save text report
            text_report_path = os.path.join(model_dir, "evaluation_report.txt")
            self.generate_evaluation_report(model_name, text_report_path)
        
        # Save model comparison
        comparison_df = self.compare_models()
        comparison_path = os.path.join(save_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        logger.info(f"All evaluation results saved to {save_dir}")
    
    def plot_model_comparison(self, save_path: str = None):
        """
        Create a bar plot comparing models' performance.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available for comparison")
            return
        
        comparison_df = self.compare_models()
        
        # Create subplot for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        metrics_to_plot = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i // 2, i % 2]
            
            if metric in comparison_df.columns:
                sns.barplot(data=comparison_df, x='model', y=metric, ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel('Model')
                ax.set_ylabel('Score')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, value in enumerate(comparison_df[metric]):
                    ax.text(j, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()

def main():
    """Main function to test the model evaluator."""
    logger.info("ModelEvaluator class ready for use")

if __name__ == "__main__":
    main()
