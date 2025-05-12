import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, ReLU, MaxPool2D, 
                                     GlobalAveragePooling2D, Dense, Dropout, Conv1D, Permute, Reshape)
from tensorflow.keras.models import Model

# Import constants from data_loader
from data_loader import N_MELS, NUM_CLASSES
# Need to calculate target_len or pass it. Let's assume we pass it during build.

def build_encoder_block(input_tensor, filters, kernel_size=(3, 3), pool_size=(2, 2), dropout=0.2):
    """Builds a single encoder block (Conv -> BN -> ReLU -> Pool -> Dropout)."""
    x = Conv2D(filters, kernel_size=kernel_size, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=pool_size)(x)
    x = Dropout(dropout)(x)
    return x

def build_cnn_model(input_shape, num_classes=NUM_CLASSES):
    """Builds the CNN model based on the diagram.

    Args:
        input_shape: Tuple representing the shape of the input spectrograms 
                     (e.g., (N_MELS, target_len, 1))
        num_classes: Number of output classes.

    Returns:
        A Keras Model instance.
    """
    inputs = Input(shape=input_shape)

    # --- Encoder Blocks --- 
    # Let's use 3 encoder blocks as suggested visually
    x = build_encoder_block(inputs, filters=32, pool_size=(2, 2)) 
    x = build_encoder_block(x, filters=64, pool_size=(2, 2)) 
    x = build_encoder_block(x, filters=128, pool_size=(2, 4)) # Pool more aggressively on time axis later

    # --- Convolution 1D over the Y axis (Frequency axis after pooling) --- 
    # Interpretation: After pooling, we have shape (batch, freq_reduced, time_reduced, channels)
    # We want to apply Conv1D across the frequency dimension.
    
    # Option 1: Global Average Pooling over time first, then Conv1D might be too simple.
    # Option 2: Reshape/Permute to apply Conv1D across frequency bands for each time step? Complex.
    
    # Let's try Option 3: Apply Global Average Pooling across the *time* dimension first.
    # This collapses the time axis, leaving (batch, freq_reduced, channels)
    # We can then treat 'freq_reduced' as the sequence length for Conv1D.
    if tf.keras.backend.image_data_format() == 'channels_last':
        # Input shape to GAP: (batch, freq_reduced, time_reduced, channels)
        # Output shape: (batch, freq_reduced, channels)
        pooled_time = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=2))(x) 
    else: # channels_first
        # Input shape to GAP: (batch, channels, freq_reduced, time_reduced)
        # Output shape: (batch, channels, freq_reduced)
        pooled_time = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=3))(x)
        # Permute to (batch, freq_reduced, channels) for Conv1D
        pooled_time = Permute((2, 1))(pooled_time)

    # Now pooled_time has shape (batch, freq_reduced, channels=128)
    # We can apply Conv1D along the 'freq_reduced' dimension.
    # Let's use a kernel size of 3 and reduce channels.
    conv1d_output = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(pooled_time)
    
    # --- Final Classification Head ---
    # Flatten the output of Conv1D
    flattened = tf.keras.layers.Flatten()(conv1d_output)
    
    x = Dense(128, activation='relu')(flattened)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    # Example usage: Print model summary
    # Requires knowing the target length from data processing
    # Let's estimate based on 4s audio (4 * 22050 / 512 hop) -> ~173 frames
    TARGET_LEN_ESTIMATE = 173 
    INPUT_SHAPE = (N_MELS, TARGET_LEN_ESTIMATE, 1)
    model = build_cnn_model(input_shape=INPUT_SHAPE)
    model.summary() 