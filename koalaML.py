import keras

class BaseModel:
    def __init__(self):
        NotImplementedError()

    def __call__(self, X):
        NotImplementedError()

class RecurrentNetwork:
    def __init__(self):
        