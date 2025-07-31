import torch
import torch.nn as nn
import torch.nn.functional as F

class AE_CIFAR(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

        self.encoder_hidden_layer = nn.Linear(in_features=input_shape, out_features=1024)
        self.encoder_middle_1 = nn.Linear(in_features=1024, out_features=512)
        self.encoder_middle = nn.Linear(in_features=512, out_features=256)
        self.encoder_output_layer = nn.Linear(in_features=256, out_features=256)  # code size

        self.decoder_hidden_layer = nn.Linear(in_features=256, out_features=256)
        self.decoder_middle1 = nn.Linear(in_features=256, out_features=512)
        self.decoder_middle = nn.Linear(in_features=512, out_features=1024)
        self.decoder_output_layer = nn.Linear(in_features=1024, out_features=input_shape)

        self.code_size = 256

    def forward(self, features):
        activation = F.leaky_relu(self.encoder_middle(
            F.leaky_relu(self.encoder_middle_1(
                F.leaky_relu(self.encoder_hidden_layer(features))
            ))
        ))
        code = F.leaky_relu(self.encoder_output_layer(activation)) 
        activation = F.leaky_relu(self.decoder_middle(
            F.leaky_relu(self.decoder_middle1(
                F.leaky_relu(self.decoder_hidden_layer(code))
            ))
        ))
        reconstructed = F.leaky_relu(self.decoder_output_layer(activation))
        return reconstructed, code
