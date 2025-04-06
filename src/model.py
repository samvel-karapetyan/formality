from transformers import AutoTokenizer, AutoModel


class Model:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, x):
        tokenized = self.tokenizer(x, return_tensors='pt', padding=True).to(self.model.device)
        out = self.model(**tokenized, output_hidden_states=True).hidden_states[-1]

        return out[:, 0].detach().cpu().numpy()

    def to(self, device):
        self.model = self.model.to(device)
        return self
