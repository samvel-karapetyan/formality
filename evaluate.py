import os
import argparse

from tqdm import tqdm

import numpy as np
import scipy.stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", required=True)
parser.add_argument("--n_repeats", required=True, type=int)


def main(model_name: str, n_repeats: int):
    data_path = os.path.join("encodings", f"{os.path.basename(model_name)}.npz")

    embeddings, labels = np.load(data_path).values()

    scores = []

    for i in tqdm(range(n_repeats)):
        x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, 
                                                            test_size=0.2, random_state=i)

        model = LogisticRegression(max_iter=10000)
        model.fit(x_train, y_train)
        scores.append(model.score(x_test, y_test))

    mean = np.mean(scores)
    sem = st.sem(scores)

    print("Score (Accuracy %):", f"{100 * mean:.2f}±{100 * sem:.2f}")

if __name__ == "__main__":
    args = parser.parse_args()

    main(model_name=args.model_name, n_repeats=args.n_repeats)
