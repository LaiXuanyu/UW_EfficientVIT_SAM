import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_segmentation(y_true, y_pred, num_classes):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=np.arange(num_classes))
    
    # Compute IoU for each class
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    
    IoU = intersection / union.astype(np.float32)
    
    # Compute precision and recall for each class
    precision = intersection / predicted_set
    recall = intersection / ground_truth_set
    
    # Compute overall accuracy
    overall_accuracy = np.sum(intersection) / np.sum(cm)
    
    # Compute mean IoU (mIoU)
    mIoU = np.mean(IoU)
    
    return {
        'IoU': IoU,
        'Precision': precision,
        'Recall': recall,
        'Overall Accuracy': overall_accuracy,
        'mIoU': mIoU
    }

# Example usage
# Generate some random predictions and ground truth labels
np.random.seed(42)
num_classes = 5  # Number of classes
y_true = np.random.randint(0, num_classes, size=(100, 100))  # Ground truth labels
y_pred = np.random.randint(0, num_classes, size=(100, 100))  # Predicted results

# Evaluate the model
metrics = evaluate_segmentation(y_true, y_pred, num_classes)

# Output the evaluation results
print("IoU per class:", metrics['IoU'])
print("Precision per class:", metrics['Precision'])
print("Recall per class:", metrics['Recall'])
print("Overall Accuracy:", metrics['Overall Accuracy'])
print("Mean IoU (mIoU):", metrics['mIoU'])
