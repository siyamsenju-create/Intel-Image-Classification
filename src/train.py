import os
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from src.preprocess import get_datasets
from src.model import build_model

def plot_history(history, output_path):
    """
    Plot and save training/validation accuracy and loss curves.
    """
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='x')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Intel Image Classifier')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_path', type=str, default='models/best_model.keras', help='Path to save best model')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained MobileNetV2 base')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 1. Load Data
    img_size = (150, 150)
    train_ds, val_ds, test_ds, class_names = get_datasets(
        data_dir=args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size
    )
    
    # Save class names for future reference (evaluation/prediction/app)
    class_map_path = os.path.join(args.results_dir, 'class_names.json')
    with open(class_map_path, 'w') as f:
        json.dump(class_names, f, indent=4)
    print(f"Class names saved to {class_map_path}")
    
    # 2. Build Model
    model = build_model(
        num_classes=len(class_names),
        input_shape=(img_size[0], img_size[1], 3),
        pretrained=args.pretrained
    )
    model.summary()
    
    # 3. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 4. Train Model
    print(f"Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )
    
    # Save training history as JSON
    history_dict_path = os.path.join(args.results_dir, 'training_history.json')
    # Convert float32 values to standard float for JSON serialization
    serializable_history = {k: [float(v) for v in l] for k, l in history.history.items()}
    with open(history_dict_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)
        
    # 5. Plot History
    plot_history(history, os.path.join(args.results_dir, 'training_curves.png'))
    print("Training pipeline completed successfully.")

if __name__ == '__main__':
    main()
