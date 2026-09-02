"""Export a HARP profile using the validated CSP checkpoint exporter."""

from __future__ import annotations

from CSP.export_csp_checkpoint_impl import main as export_csp_main


def main() -> int:
    """Translate the HARP command to the shared CSP exporter."""

    return export_csp_main()


if __name__ == "__main__":
    raise SystemExit(main())
