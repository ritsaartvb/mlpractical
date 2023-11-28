import unittest
import torch
from model_architectures import BNConvolutionalProcessingBlock, BNConvolutionalDimensionalityReductionBlock, BNRCConvolutionalProcessingBlock

class TestBlocks(unittest.TestCase):

    def test_BNConvolutionalProcessingBlock(self):
        # Create a test input tensor
        input_tensor = torch.randn(1, 3, 64, 64)  # Example shape: (batch_size, channels, height, width)

        # Initialize the block
        block = BNConvolutionalProcessingBlock(input_shape=input_tensor.shape, 
                                                     num_filters=16, 
                                                     kernel_size=3, 
                                                     padding=1, 
                                                     bias=False, 
                                                     dilation=1)

        # Forward propagation
        output = block(input_tensor)

        # Check if output has the right shape
        self.assertEqual(output.shape, torch.Size([1, 16, 64, 64]))  # The expected output shape
    
    def test_BNRCConvolutionalProcessingBlock(self):
        # Create a test input tensor
        input_tensor = torch.randn(1, 3, 64, 64)  # Example shape: (batch_size, channels, height, width)

        # Initialize the block
        block = BNRCConvolutionalProcessingBlock(input_shape=input_tensor.shape, 
                                                     num_filters=3, 
                                                     kernel_size=3, 
                                                     padding=1, 
                                                     bias=False, 
                                                     dilation=1)

        # Forward propagation
        output = block(input_tensor)

        # Check if output has the right shape
        self.assertEqual(output.shape, torch.Size([1, 3, 64, 64]))  # The expected output shape

    def test_BNConvolutionalDimensionalityReductionBlock(self):
        # Create a test input tensor
        input_tensor = torch.randn(1, 3, 64, 64)  # Example shape: (batch_size, channels, height, width)

        # Initialize the block
        block = BNConvolutionalDimensionalityReductionBlock(input_shape=input_tensor.shape, 
                                                                  num_filters=16, 
                                                                  kernel_size=3, 
                                                                  padding=1, 
                                                                  bias=False, 
                                                                  dilation=1, 
                                                                  reduction_factor=2)

        # Forward propagation
        output = block(input_tensor)

        # Check if output has the right shape
        # The output height and width should be reduced due to the average pooling layer
        self.assertEqual(output.shape, torch.Size([1, 16, 32, 32]))  # The expected output shape

if __name__ == '__main__':
    unittest.main()

