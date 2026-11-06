from pipeline import run_link_prediction_training
from utils.load_configs import get_link_prediction_args


def main():
    run_link_prediction_training(get_link_prediction_args())


if __name__ == "__main__":
    main()
