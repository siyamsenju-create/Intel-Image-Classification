import tensorflow as tf
from tensorflow.keras import layers, Model
from src.preprocess import get_data_augmentation

def build_model(num_classes=6, input_shape=(150, 150, 3), pretrained=True, trainable_base_layers=0):
    """
    Build the neural network model for image classification.
    
    Args:
        num_classes (int): Number of target classes.
        input_shape (tuple): Dimension of input images.
        pretrained (bool): Whether to use a pretrained MobileNetV2 model.
        trainable_base_layers (int): Number of layers at the end of the base model to unfreeze.
        
    Returns:
        model: Compiled tf.keras.Model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1. Data Augmentation
    x = get_data_augmentation()(inputs)
    
    if pretrained:
        # 2. Rescale input for MobileNetV2 (expects [-1, 1])
        x = layers.Rescaling(1./127.5, offset=-1)(x)
        
        # 3. Base pretrained model (MobileNetV2)
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model by default
        base_model.trainable = False
        
        # If fine-tuning is requested, unfreeze the last N layers
        if trainable_base_layers > 0:
            base_model.trainable = True
            # Freeze all layers except the last N
            for layer in base_model.layers[:-trainable_base_layers]:
                layer.trainable = False
                
        x = base_model(x, training=False)
    else:
        # 2. Rescale input for custom CNN (expects [0, 1])
        x = layers.Rescaling(1./255)(x)
        
        # 3. Custom CNN Architecture from scratch
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
    # 4. Dense Classifier Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
