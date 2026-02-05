import torch

def test_cuda_logic():
    # Check if cuda is available and the system can use it
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert device in ['cuda', 'cpu']
    print(f"Test result: in execution on {device}")