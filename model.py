import torch
import torch.nn as nn
import torch.nn.functional as F

# Import constants from data_loader
from data_loader import N_MELS, NUM_CLASSES
# Need to calculate target_len or pass it. Let's assume we pass it during build.

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), pool_size=(2, 2), dropout_p=0.2):
        super().__init__()
        # PyTorch Conv2d default padding is 0. 'same' in TF means padding to keep spatial dims.
        # For kernel_size=3, padding=1 achieves this.
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_size=pool_size)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class AudioCNN(nn.Module):
    def __init__(self, n_mels, num_classes, target_len_estimate_for_fc_input_calc=173): # Added target_len for FC input calc
        super().__init__()
        self.n_mels = n_mels
        self.num_classes = num_classes

        # --- Encoder Blocks ---
        # Input shape: (batch, channels, freq, time) for PyTorch
        # Initial input has 1 channel (mono spectrogram)
        self.encoder1 = EncoderBlock(in_channels=1, out_channels=32, pool_size=(2, 2))
        self.encoder2 = EncoderBlock(in_channels=32, out_channels=64, pool_size=(2, 2))
        self.encoder3 = EncoderBlock(in_channels=64, out_channels=128, pool_size=(2, 4)) # Pool more on time

        # --- Convolution 1D over the Y axis (Frequency axis after pooling) ---
        # After pooling, shape is (batch, 128, freq_reduced, time_reduced)
        # We need to calculate freq_reduced based on n_mels and pooling.
        # n_mels -> pool(2,2) -> n_mels/2 -> pool(2,2) -> n_mels/4 -> pool(2,4) -> n_mels/8
        # The Conv1D was applied after reducing the time axis.
        # TF: tf.reduce_mean(x, axis=2) with (batch, freq_reduced, time_reduced, channels)
        # PyTorch: x.mean(dim=3) with (batch, channels, freq_reduced, time_reduced)

        # The output of encoder3 will have 128 channels.
        self.conv1d = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1) # padding='same'
        self.relu_conv1d = nn.ReLU() # Added ReLU after Conv1D as in original TF model
        
        # --- Final Classification Head ---
        self.flatten = nn.Flatten()

        # To define the input size for the first Linear layer, we need to calculate
        # the output shape after conv1d.
        # This requires knowing the input dimensions to the model (n_mels, target_len)
        # Let's calculate it dynamically or pass an estimate
        
        # Calculate freq_reduced and time_reduced after encoder blocks
        # Input: (1, n_mels, target_len_estimate_for_fc_input_calc)
        # After enc1: n_mels/2, target_len/2
        # After enc2: n_mels/4, target_len/4
        # After enc3: n_mels/8, target_len/16
        
        # This is a placeholder calculation. A more robust way is to do a dummy forward pass.
        # Or, ensure target_len_estimate_for_fc_input_calc is accurately passed.
        # The TF model applies Conv1D to (batch, freq_reduced, channels=128) where freq_reduced is the sequence.
        # Output of Conv1D is (batch, new_channels=64, freq_reduced) due to padding='same'.
        
        # freq_after_encoders = n_mels // 8
        # fc1_input_features = 64 * freq_after_encoders # This matches the TF model's structure

        # Simulating the forward pass dimensions to get the flattened size
        # Assume input_shape is (batch, 1, n_mels, target_len_estimate_for_fc_input_calc)
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.n_mels, target_len_estimate_for_fc_input_calc)
            dummy_x = self.encoder1(dummy_input)
            dummy_x = self.encoder2(dummy_x)
            dummy_x = self.encoder3(dummy_x)
            
            # TF model: (batch, freq_reduced, time_reduced, channels) -> reduce_mean(axis=2) -> (batch, freq_reduced, channels)
            # PyTorch: (batch, channels, freq_reduced, time_reduced) -> mean(dim=3) -> (batch, channels, freq_reduced)
            pooled_time_dummy = dummy_x.mean(dim=3) # (batch, 128, freq_reduced_dummy)
            
            # PyTorch Conv1D expects (batch, channels, length)
            # Here, channels are 128, length is freq_reduced_dummy
            conv1d_out_dummy = self.conv1d(pooled_time_dummy) # (batch, 64, freq_reduced_dummy)
            flattened_dummy = self.flatten(conv1d_out_dummy)
            fc1_input_features = flattened_dummy.shape[1]

        self.fc1 = nn.Linear(fc1_input_features, 128)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)
        # Softmax is usually part of the loss function (e.g., CrossEntropyLoss)
        # self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        # Input x shape: (batch, n_mels, time) - typical for audio
        # Add channel dimension: (batch, 1, n_mels, time)
        x = x.unsqueeze(1)

        # --- Encoder Blocks ---
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x) # Shape: (batch, 128, freq_reduced, time_reduced)

        # --- Time Reduction & Conv1D ---
        # In TF: reduce_mean on time axis (axis=2 for channels_last) -> (batch, freq_reduced, channels)
        # In PyTorch: input is (batch, channels, freq, time), so reduce on dim=3 (time)
        x = x.mean(dim=3) # Shape: (batch, 128, freq_reduced)
        
        # Conv1D expects (batch, channels, length)
        # Here, channels=128 (from encoder3), length=freq_reduced
        x = self.conv1d(x)
        x = self.relu_conv1d(x) # Shape: (batch, 64, freq_reduced)

        # --- Final Classification Head ---
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        # x = self.softmax(x) # Apply softmax if not included in loss
        return x

if __name__ == '__main__':
    # Example usage: Print model summary (more or less)
    # These values should come from data_loader.py or config.py eventually
    N_MELS_EXAMPLE = 128  # from data_loader.py
    NUM_CLASSES_EXAMPLE = 10 # from data_loader.py
    TARGET_LEN_ESTIMATE = 173 # from original model.py example
    
    # PyTorch expects (batch, channels, height, width) or (batch, features) for Linear
    # Our raw input will be (batch, N_MELS_EXAMPLE, TARGET_LEN_ESTIMATE)
    # It gets unsqueezed to (batch, 1, N_MELS_EXAMPLE, TARGET_LEN_ESTIMATE) in forward()
    
    model = AudioCNN(n_mels=N_MELS_EXAMPLE, num_classes=NUM_CLASSES_EXAMPLE, target_len_estimate_for_fc_input_calc=TARGET_LEN_ESTIMATE)
    
    print(model)
    
    # Test with a dummy input
    batch_size = 4
    # Input spectrograms (batch, n_mels, time_frames)
    dummy_input_spectrogram = torch.randn(batch_size, N_MELS_EXAMPLE, TARGET_LEN_ESTIMATE) 
    
    print(f"\nInput shape to model: {dummy_input_spectrogram.shape}")
    
    model.eval() # Set model to evaluation mode for inference
    with torch.no_grad(): # Disable gradient calculations
        output = model(dummy_input_spectrogram)
    
    print(f"Output shape from model: {output.shape}") # Expected: (batch_size, NUM_CLASSES_EXAMPLE)
    assert output.shape == (batch_size, NUM_CLASSES_EXAMPLE)
    print("Model instantiation and dummy forward pass successful.")

    # For a more detailed summary like Keras model.summary(), you might use torchinfo
    # try:
    #     from torchinfo import summary
    #     # For torchinfo, provide input_size excluding batch_dim, but model adds channel dim
    #     # The model's forward unsqueezes(1), so input to torchinfo should represent that
    #     # summary(model, input_size=(1, N_MELS_EXAMPLE, TARGET_LEN_ESTIMATE))
    #     # OR, if your model expects raw (N_MELS, TARGET_LEN) and handles unsqueeze internally:
    #     summary(model, input_size=(N_MELS_EXAMPLE, TARGET_LEN_ESTIMATE), batch_dim=0) # batch_dim=0 to handle the first dim as batch
    # except ImportError:
    #     print("Install torchinfo for a Keras-like model summary: pip install torchinfo") 