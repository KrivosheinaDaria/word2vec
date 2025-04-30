import torch.nn as nn

class Net(nn.Module):
    def __init__(self, vector_size, dictionary_size):
        super(Net, self).__init__()
        self.vector_size = vector_size
        self.dictionary_size = dictionary_size

        self.pipe = nn.Sequential(
            nn.Linear(self.vector_size, self.vector_size),

            nn.Linear(self.vector_size, 500),
            nn.ReLU(),

            nn.Linear(500, self.dictionary_size))

    def forward(self, x):
        x = x.float()
        return self.pipe(x)