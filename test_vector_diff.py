import torch
from vector_diff import VectorDiffusion
from encoder import Encoder

def test_encoder():
    """
    Test that the modified Encoder can handle 1D input correctly.
    """
    print("\n=== Testing Encoder ===")
    
    try:
        # Create an encoder with the correct parameters
        encoder = Encoder(
            input_dim=6000,  # Length of xPDF y-values
            hidden_dim=64,
            output_dim=32,
            num_layers=2,
            type='MLP'
        )
        
        # Create fake xPDF y-values with shape [batch_size, sequence_length]
        batch_size = 2
        y_values = torch.randn(batch_size, 6000)
        
        # Test forward pass
        print("Testing encoder forward pass...")
        output = encoder(y_values)
        print(f"Encoder output shape: {output.shape}")
        assert output.shape == (batch_size, 32), f"Expected shape {(batch_size, 32)}, got {output.shape}"
        
        print("Encoder test passed!")
        return True
    except Exception as e:
        print(f"Encoder test failed with error: {e}")
        return False

def test_vector_diffusion():
    """
    Test that the VectorDiffusion model can handle xPDF data correctly.
    """
    print("\n=== Testing VectorDiffusion ===")
    
    try:
        # Create a model with the correct parameters
        model = VectorDiffusion(
            in_channels=6000,  # Length of xPDF y-values
            hidden_channels=64,
            out_channels=30,  # 10 atoms with 3 coordinates each
            T=10  # Small number of steps for testing
        )
        
        # Create fake xPDF data with shape [batch_size, 2, 6000]
        batch_size = 2
        xpdf = torch.randn(batch_size, 2, 6000)
        
        # Create fake atom positions with shape [batch_size, out_channels]
        positions = torch.randn(batch_size, 30)
        
        # Test forward diffusion
        print("Testing forward diffusion...")
        t = torch.randint(1, model.T, (batch_size,))
        epsilon = torch.randn_like(positions)
        noisy_positions = model.forward_diffusion(positions, t, epsilon)
        print(f"Noisy positions shape: {noisy_positions.shape}")
        assert noisy_positions.shape == positions.shape, f"Expected shape {positions.shape}, got {noisy_positions.shape}"
        
        # Test reverse diffusion
        print("Testing reverse diffusion...")
        y_values = xpdf[:, 1, :]  # Extract y-values
        noise = torch.randn_like(positions)
        t_tensor = torch.full((batch_size,), 5, dtype=torch.long)
        denoised_positions = model.reverse_diffusion(positions, t_tensor, noise, y_values)
        print(f"Denoised positions shape: {denoised_positions.shape}")
        assert denoised_positions.shape == positions.shape, f"Expected shape {positions.shape}, got {denoised_positions.shape}"
        
        # Test loss calculation
        print("Testing loss calculation...")
        loss = model.loss(positions, xpdf)
        print(f"Loss value: {loss.item()}")
        
        # Test sampling
        print("Testing sampling...")
        samples = model.sample((batch_size, 30), xpdf)
        print(f"Samples shape: {samples.shape}")
        assert samples.shape == positions.shape, f"Expected shape {positions.shape}, got {samples.shape}"
        
        # Test forward pass (full model)
        print("Testing forward pass...")
        output = model(xpdf)
        print(f"Output shape: {output.shape}")
        assert output.shape == positions.shape, f"Expected shape {positions.shape}, got {output.shape}"
        
        print("VectorDiffusion test passed!")
        return True
    except Exception as e:
        import traceback
        print(f"VectorDiffusion test failed with error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    encoder_success = test_encoder()
    diffusion_success = test_vector_diffusion()
    
    if encoder_success and diffusion_success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed. Please check the error messages above.") 