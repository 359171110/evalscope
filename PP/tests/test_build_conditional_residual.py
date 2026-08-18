from __future__ import annotations

import torch

from PP.build_conditional_residual import conditional_residual_order, functional_product_kernel


def test_functional_product_kernel_is_psd() -> None:
    responses = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    down = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    kernel = functional_product_kernel(responses, down)

    assert torch.linalg.eigvalsh(kernel).min() >= -1.0e-5


def test_conditional_residual_matches_direct_schur_choice() -> None:
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    kernel = features @ features.transpose(0, 1)
    importance_order = torch.tensor([0, 1, 2])
    pseudo_order = torch.tensor([0, 1, 2])

    order, _ = conditional_residual_order(
        kernel,
        importance_order,
        pseudo_order,
        retained_channels=2,
        protected_channels=1,
        ridge_relative=1.0e-6,
    )

    assert order[:2].tolist() == [0, 2]
    assert sorted(order.tolist()) == [0, 1, 2]


def test_importance_weighting_suppresses_weak_unique_channel() -> None:
    responses = torch.eye(3)
    down = torch.eye(3)
    importance = torch.tensor([1.0, 0.8, 1.0e-6])
    kernel = functional_product_kernel(responses, down, importance=importance)

    order, _ = conditional_residual_order(
        kernel,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 2]),
        retained_channels=2,
        protected_channels=0,
    )

    assert order[:2].tolist() == [0, 1]