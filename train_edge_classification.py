from pipeline import run_edge_classification_training
from utils.load_configs import get_edge_classification_args


def main():
    run_edge_classification_training(get_edge_classification_args())


if __name__ == "__main__":
    main()
