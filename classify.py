import argparse

from experiments.experiments_eager import classification_experiments


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-e", "--experiments", type=str, nargs="+", help="Experiments to run"
    )
    argParser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Dataset to use for experiment. gene or tweets.",
        default="tweets",
    )
    argParser.add_argument("--wandb", action=argparse.BooleanOptionalAction)

    args = argParser.parse_args()

    if args.experiments == None:
        classification_experiments(use_wandb=args.wandb, dataset=args.dataset)
    else:
        classification_experiments(
            use_wandb=args.wandb, experiments=args.experiments, dataset=args.dataset
        )


if __name__ == "__main__":
    main()
