import random


class ReplayMemory:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.batch_size = batch_size

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position % self.capacity] = experience
        self.position += 1

    def sample(self):
        out = random.sample(self.memory, self.batch_size)
        return list(map(list, zip(*out)))

    def can_sample(self):
        return len(self.memory) >= self.batch_size

    def episode_sample(self):
        out = self.memory
        return list(map(list, zip(*out)))

    def __len__(self):
        return len(self.memory)
