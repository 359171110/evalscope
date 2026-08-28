"""CSP checkpoint export entrypoint."""

from CSP.export_csp_checkpoint_impl import main as _implementation_main


def main() -> int:
    """Run the CSP checkpoint exporter implementation."""

    return _implementation_main()


if __name__ == "__main__":
    raise SystemExit(main())
