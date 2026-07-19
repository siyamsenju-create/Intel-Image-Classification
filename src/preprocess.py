import os
import tensorflow as tf
from tensorflow.keras import layers

def get_datasets(data_dir, img_size=(150, 150), batch_size=32, val_split=0.2, seed=42):
    """
    Load train, validation, and test datasets from directory.
    
    Args:
        data_dir (str): Path to data directory containing 'train' and 'test' folders.
        img_size (tuple): Target image height and width.
        batch_size (int): Size of batches.
        val_split (float): Validation split fraction from the training set.
        seed (int): Random seed for reproducibility.
        
    Returns:
        train_ds, val_ds, test_ds: Preprocessed tf.data.Dataset objects.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    print("Loading training dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    print("Loading validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    print("Loading test dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    
    # Class names are extracted from the directories
    class_names = train_ds.class_names
    print(f"Loaded class names: {class_names}")
    
    # Optimize datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names

def get_data_augmentation():
    """
    Create a data augmentation sequential block.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
    ])
    return data_augmentation
