from pipeline import run_link_retrieval_evaluation
from utils.load_configs import get_link_prediction_args


def main():
    run_link_retrieval_evaluation(get_link_prediction_args())


if __name__ == "__main__":
    main()
