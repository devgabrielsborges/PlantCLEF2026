import torch

from src.models.dann import grad_reverse


def test_grl_forward():
    """Verify GRL acts as identity during forward pass."""
    x = torch.randn(2, 5, requires_grad=True)
    alpha = 0.5
    y = grad_reverse(x, alpha)

    # Forward pass should be identity
    assert torch.equal(x, y)


def test_grl_backward():
    """Verify GRL reverses and scales gradient during backward pass."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    alpha = 0.5

    # y = x * 1.0 (identity forward)
    y = grad_reverse(x, alpha)

    # Loss = sum(y)
    loss = y.sum()
    loss.backward()

    # Gradient of loss wrt y is [1.0, 1.0]
    # GRL backward should negate and scale it by alpha: -0.5 * [1.0, 1.0] = [-0.5, -0.5]
    expected_grad = torch.tensor([-0.5, -0.5])
    assert torch.allclose(x.grad, expected_grad)


def test_grl_zero_alpha():
    """Verify GRL with alpha=0.0 results in zero gradient."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    alpha = 0.0
    y = grad_reverse(x, alpha)
    loss = y.sum()
    loss.backward()

    expected_grad = torch.tensor([0.0, 0.0])
    assert torch.allclose(x.grad, expected_grad)


def test_dann_forward():
    """Verify DANN model returns correct output shapes."""
    from src.models.dann import DANN

    num_classes = 10
    backbone = "resnet18"  # Smaller backbone for fast test
    model = DANN(backbone, num_classes)

    x = torch.randn(2, 3, 224, 224)
    alpha = 0.5
    class_out, domain_out = model(x, alpha)

    # Class output should be (batch_size, num_classes)
    assert class_out.shape == (2, 10)
    # Domain output should be (batch_size, 1)
    assert domain_out.shape == (2, 1)
    # Domain output should be in range [0, 1] due to Sigmoid
    assert torch.all(domain_out >= 0) and torch.all(domain_out <= 1)
