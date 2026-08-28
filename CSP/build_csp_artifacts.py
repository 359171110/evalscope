"""CSP artifact builder entrypoint."""

from CSP.build_csp_artifacts_impl import main as _implementation_main


def main() -> int:
    """Run the CSP artifact builder implementation."""

    return _implementation_main()


if __name__ == "__main__":
    raise SystemExit(main())
