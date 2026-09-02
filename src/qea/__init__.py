import argparse
from pathlib import Path

from .config import read_config


def create_argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="QEA",
        description="A quantum evolutionary algorithm",
    )
    parser.add_argument("-c", "--config", type=Path, required=False)
    return parser.parse_args()


def main() -> None:

    args = create_argument_parser()
    config = read_config(args.config)

    print(config)
