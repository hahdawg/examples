from os.path import join
import pickle
import time


class Car:

    def __init__(self, model: str, price: int):
        self.model = model
        self.price = price
        self.save_path = join("/tmp", f"{self.model}_{self.price}.p")

    def speak(self) -> str:
        return f"I am a {self.model} and I'm worth {self.price}."

    def save(self):  # pylint: disable=R0201
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)


def add_slow(x, y):
    time.sleep(3)
    return x + y


def add_slow_and_multiply(x, y, c):
    s = add_slow(x, y)
    m = c*s
    return m
