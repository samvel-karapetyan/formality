import argparse
import os
import numpy as np
from tqdm import tqdm

from src import DataModule, Model

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--num_workers", required=True, type=int)


def main(model_name: str, device: str, batch_size: int, num_workers: int):
    datamodule = DataModule(batch_size, num_workers)
    model = Model(model_name).to(device)
    loader = datamodule.test_dataloader()

    embeddings = []
    labels = []

    for batch in tqdm(loader):
        x, y = batch
        
        emb = model(x)

        embeddings.append(emb)
        labels.extend(y)

    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)

    np.savez(os.path.join("encodings", f"{os.path.basename(model_name)}.npz"), 
            embeddings=embeddings, labels=labels)

if __name__ == "__main__":
    args = parser.parse_args()

    main(model_name=args.model_name, device=args.device, 
         batch_size=args.batch_size, num_workers=args.num_workers)