import logging
import os
import sys
from typing import Any, Dict

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sentiment_analysis.log')
        ]
    )

def ensure_dir(directory: str) -> None:
    """
    Ensure that a directory exists.
    
    Args:
        directory: Directory path to create if it doesn't exist
    """
    os.makedirs(directory, exist_ok=True)

def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics to save
        filepath: Path to save the metrics
    """
    import json
    
    ensure_dir(os.path.dirname(filepath))
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {filepath}")

def load_metrics(filepath: str) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.
    
    Args:
        filepath: Path to the metrics file
        
    Returns:
        Dictionary of metrics
    """
    import json
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    logger.info(f"Metrics loaded from {filepath}")
    return metrics

def get_project_root() -> str:
    """
    Get the root directory of the project.
    
    Returns:
        Path to the project root directory
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def validate_data_structure(df, required_columns: list = None) -> bool:
    """
    Validate that a DataFrame has the required structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    if required_columns is None:
        required_columns = ['review', 'sentiment']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for null values in required columns
    null_columns = [col for col in required_columns if df[col].isnull().any()]
    if null_columns:
        raise ValueError(f"Null values found in required columns: {null_columns}")
    
    logger.info("Data structure validation passed")
    return True

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def calculate_text_statistics(texts) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of texts.
    
    Args:
        texts: List or Series of text strings
        
    Returns:
        Dictionary with text statistics
    """
    import numpy as np
    
    if len(texts) == 0:
        return {}
    
    # Calculate text lengths
    text_lengths = [len(str(text).split()) for text in texts]
    
    stats = {
        'mean_length': np.mean(text_lengths),
        'median_length': np.median(text_lengths),
        'min_length': np.min(text_lengths),
        'max_length': np.max(text_lengths),
        'std_length': np.std(text_lengths),
        'total_texts': len(texts)
    }
    
    return stats

def create_confusion_matrix_plot(y_true, y_pred, labels=None, save_path=None):
    """
    Create and optionally save a confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def create_roc_curve_plot(y_true, y_proba, save_path=None):
    """
    Create and optionally save an ROC curve plot.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    plt.show()

def main():
    """Main function to test utility functions."""
    # Test logging setup
    setup_logging("INFO")
    logger.info("Testing utility functions")
    
    # Test directory creation
    ensure_dir("test_dir")
    logger.info("Directory creation test passed")
    
    # Test time formatting
    print(format_time(45))
    print(format_time(120))
    print(format_time(7200))
    
    # Test text statistics
    sample_texts = ["This is a test", "Another test", "Short", "This is a much longer text for testing"]
    stats = calculate_text_statistics(sample_texts)
    print(f"Text statistics: {stats}")

if __name__ == "__main__":
    main()
