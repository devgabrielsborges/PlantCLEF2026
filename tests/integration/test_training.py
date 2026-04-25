import torch
import torch.nn as nn
import torch.optim as optim

from src.models.dann import DANN


def test_training_step():
    """Verify one full adversarial training step logic."""
    device = torch.device("cpu")
    num_classes = 5
    model = DANN("resnet18", num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    alpha = 0.5
    batch_size = 2

    # 1. Dummy source data
    s_img = torch.randn(batch_size, 3, 224, 224).to(device)
    s_cls_label = torch.randint(0, num_classes, (batch_size,)).to(device)
    s_dom_label = torch.zeros(batch_size, 1).to(device)

    # 2. Dummy target data
    t_img = torch.randn(batch_size, 3, 224, 224).to(device)
    t_dom_label = torch.ones(batch_size, 1).to(device)

    # 3. Training step
    model.train()
    optimizer.zero_grad()

    # Source pass
    s_cls_out, s_dom_out = model(s_img, alpha)
    loss_s_cls = criterion_class(s_cls_out, s_cls_label)
    loss_s_dom = criterion_domain(s_dom_out, s_dom_label)

    # Target pass
    _, t_dom_out = model(t_img, alpha)
    loss_t_dom = criterion_domain(t_dom_out, t_dom_label)

    total_loss = loss_s_cls + loss_s_dom + loss_t_dom
    total_loss.backward()

    # Check if gradients are computed for feature extractor
    has_grad = False
    for param in model.feature_extractor.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "Feature extractor should have gradients after backward pass."

    # Update weights
    optimizer.step()

    assert total_loss.item() > 0
