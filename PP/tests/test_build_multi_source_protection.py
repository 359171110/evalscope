import torch

from PP.build_multi_source_protection import parse_source_spec, select_quota_protection


def test_two_source_quota_fill_skips_overlap() -> None:
    pp_order = torch.arange(10)
    pwrp_order = torch.tensor([0, 1, 2, 3, 9, 8, 7, 6, 5, 4])

    protected, diagnostics = select_quota_protection(
        [("PP", 4, pp_order), ("PWRP", 3, pwrp_order)]
    )

    assert protected.tolist() == [0, 1, 2, 3, 9, 8, 7]
    assert protected.unique().numel() == 7
    assert diagnostics["PP_scan_depth"] == 4.0
    assert diagnostics["PWRP_scan_depth"] == 7.0


def test_three_source_quota_fill_preserves_declared_order() -> None:
    pp_order = torch.arange(10)
    esp_order = torch.tensor([0, 1, 9, 8, 7, 6, 5, 4, 3, 2])
    pwrp_order = torch.tensor([9, 8, 0, 1, 2, 7, 6, 5, 4, 3])

    protected, diagnostics = select_quota_protection(
        [("PP", 3, pp_order), ("ESP", 2, esp_order), ("PWRP", 2, pwrp_order)]
    )

    assert protected.tolist() == [0, 1, 2, 9, 8, 7, 6]
    assert protected.unique().numel() == 7
    assert diagnostics["protected_channels"] == 7.0


def test_source_spec_parses_name_quota_and_path() -> None:
    name, quota, path = parse_source_spec("ESP=25=/tmp/esp.pt")

    assert name == "ESP"
    assert quota == 25
    assert path.name == "esp.pt"