import argparse

from torch_pharma.utils.logging import get_pylogger, setup_logging

log = get_pylogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="torch_pharma evaluation CLI")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    setup_logging(level=args.log_level, log_file=args.log_file, run_name="evaluate")
    log.info("Starting evaluation CLI (configure experiment script for full workflow)")


if __name__ == "__main__":
    main()
