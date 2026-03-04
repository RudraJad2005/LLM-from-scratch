class Trainer:
    def __init__(self, model):
        self.model = model

    def train_step(self, batch):
        _ = batch
        return {"loss": 0.0}
