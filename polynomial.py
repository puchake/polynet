""" This script is the main entry point for this application. """


import argparse

from commands import train, estimate


def positive_int(value):
    """
    Check if "value" variable contains positive integer and return it. If
    "value" does not contain positive integer then exception is raised.
    """
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("{} is not a positive integer value"
                                         .format(value))
    return int_value


def main():
    # Create main parser.
    args_parser = argparse.ArgumentParser(description="")
    subparsers = args_parser.add_subparsers(
        help="Choose a command which will be executed by the program. Typing -h"
             " after the command will show further help.",
        dest="command"
    )

    # Create subparser for train command.
    train_parser = subparsers.add_parser(
        "train",
        help="Train a neural network to find coefficients of a polynomial which"
             " fits the dataset the best."
    )
    train_parser.add_argument(
        "polynomial_degree", type=positive_int,
        help="Maximal degree of the polynomial which neural network will try to"
             " find."
    )
    train_parser.add_argument(
        "path_to_csv", type=str,
        help="Path to a .csv file which contains dataset of points."
    )

    # Create subparser for estimate command.
    estimate_parser = subparsers.add_parser(
        "estimate",
        help="Use previously trained neural network to estimate y value of some"
             " x."
    )
    estimate_parser.add_argument("x", type=float,
                                 help="x value for which y will be estimated.")

    args = args_parser.parse_args()
    if args.command == "train":
        coeffs = train(args.path_to_csv, args.polynomial_degree)
        print(coeffs)
    elif args.command == "estimate":
        y = estimate(args.x)
        print(y)


if __name__ == "__main__":
    main()
