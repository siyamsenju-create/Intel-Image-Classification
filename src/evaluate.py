import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

def load_images_from_dir(class_dir, class_idx, img_size, x_data, y_data):
    """Load all images from a class directory into arrays."""
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [f for f in os.listdir(class_dir) 
             if os.path.splitext(f.lower())[1] in valid_exts]
    
    for fname in files:
        img_path = os.path.join(class_dir, fname)
        try:
            img = Image.open(img_path).convert('RGB').resize(img_size)
            x_data.append(np.array(img, dtype=np.float32))
            y_data.append(class_idx)
        except Exception as e:
            print(f"  Warning: skipping {fname}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Intel Image Classifier')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='models/best_model.keras', help='Path to saved model')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for prediction')
    
    args = parser.parse_args()
    
    # Ensure results directory exists
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 1. Load class names
    class_map_path = os.path.join(args.results_dir, 'class_names.json')
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"Class names not found at {class_map_path}. Train first.")
        
    with open(class_map_path, 'r') as f:
        class_names = json.load(f)
    print(f"Loaded class names: {class_names}")
    
    # 2. Load test dataset directly with PIL (no tf.data to avoid hangs)
    test_dir = os.path.join(args.data_dir, 'test')
    img_size = (150, 150)
    
    print("Loading test images directly with PIL...")
    x_data, y_data = [], []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: directory not found: {class_dir}")
            continue
        n_before = len(x_data)
        load_images_from_dir(class_dir, class_idx, img_size, x_data, y_data)
        print(f"  Loaded {len(x_data) - n_before} images for class '{class_name}'")
    
    x_data = np.array(x_data, dtype=np.float32)
    y_true = np.array(y_data, dtype=np.int32)
    print(f"Total test images: {len(x_data)}")
    
    # 3. Load model
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    
    # 4. Batch-by-batch prediction (no tf.data pipeline)
    print("Running predictions...")
    y_pred_probs_list = []
    n = len(x_data)
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch = x_data[start:end]
        probs = model(batch, training=False).numpy()
        y_pred_probs_list.append(probs)
        print(f"  Processed {end}/{n} images", flush=True)
    
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=0) if len(y_pred_probs.shape) == 1 else np.argmax(y_pred_probs, axis=1)
    
    # Compute metrics manually
    one_hot = np.zeros((len(y_true), len(class_names)), dtype=np.float32)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    eps = 1e-15
    y_clipped = np.clip(y_pred_probs, eps, 1.0 - eps)
    test_loss = float(-np.mean(np.sum(one_hot * np.log(y_clipped), axis=1)))
    test_accuracy = float(np.mean(y_true == y_pred))
    
    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    
    # 5. Classification report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_str  = classification_report(y_true, y_pred, target_names=class_names)
    
    print("\nClassification Report:\n")
    print(report_str)
    
    with open(os.path.join(args.results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report_str)
    print(f"Classification report saved.")
    
    with open(os.path.join(args.results_dir, 'metrics.json'), 'w') as f:
        json.dump({'test_loss': test_loss, 'test_accuracy': test_accuracy,
                   'classification_report': report_dict}, f, indent=4)
    print("Metrics JSON saved.")
    
    # 6. Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Intel Image Classification')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    cm_path = os.path.join(args.results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    print("Evaluation completed successfully.")

if __name__ == '__main__':
    main()
