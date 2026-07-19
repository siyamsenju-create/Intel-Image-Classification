import os
import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image

def predict_image(image_path, model, class_names, img_size=(150, 150)):
    """
    Run inference on a single image.
    
    Args:
        image_path (str): Path to image file.
        model (tf.keras.Model): Trained classification model.
        class_names (list): List of class names.
        img_size (tuple): Target size for input image.
        
    Returns:
        predicted_class (str): Class with highest probability.
        confidence (float): Probability of the predicted class.
        prob_dict (dict): Dictionary mapping class names to probabilities.
    """
    # Load and preprocess the image
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis
    
    # Run model prediction using direct call (avoids tf.data threading deadlock)
    predictions = model(img_array, training=False).numpy()
    probabilities = predictions[0]
    
    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])
    
    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    return predicted_class, confidence, prob_dict

def main():
    parser = argparse.ArgumentParser(description='Inference with Intel Image Classifier')
    parser.add_argument('--image_path', type=str, required=True, help='Path to target image file')
    parser.add_argument('--model_path', type=str, default='models/best_model.keras', help='Path to saved model')
    parser.add_argument('--class_names_path', type=str, default='results/class_names.json', help='Path to class names JSON')
    
    args = parser.parse_args()
    
    # Verify file paths
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found at {args.image_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}. Please train the model first.")
    if not os.path.exists(args.class_names_path):
        raise FileNotFoundError(f"Class names file not found at {args.class_names_path}. Please train the model first.")
        
    # Load class names
    with open(args.class_names_path, 'r') as f:
        class_names = json.load(f)
        
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    
    # Run prediction
    print(f"Predicting for image {args.image_path}...")
    predicted_class, confidence, prob_dict = predict_image(args.image_path, model, class_names)
    
    print("\nPrediction Results:")
    print("-" * 30)
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence:      {confidence:.2%}")
    print("\nClass Probabilities:")
    for cls, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:<12} : {prob:.2%}")
    print("-" * 30)

if __name__ == '__main__':
    main()
