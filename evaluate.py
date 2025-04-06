import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", required=True)


def main(model_name: str):
    data_path = os.path.join("encodings", f"{os.path.basename(model_name)}.npz")

    embeddings, labels = np.load(data_path).values()

    x_train, x_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2)

    model = LogisticRegression(max_iter=10000)
    model.fit(x_train, y_train)

    print("Score:", model.score(x_test, y_test))

if __name__ == "__main__":
    args = parser.parse_args()

    main(model_name=args.model_name)