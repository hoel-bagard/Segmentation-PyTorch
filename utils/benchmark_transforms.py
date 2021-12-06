import argparse
from time import time


class PowerOfDict(object):
    """Returns number to the power of N."""

    def __init__(self, n: int):
        self.n = n

    def __call__(self, sample):
        nb, label = sample["nb"], sample["label"]
        nb = nb ** self.n
        return {"nb": nb, "label": label}


class PowerOf(object):
    """Returns number to the power of N."""

    def __init__(self, n: int):
        self.n = n

    def __call__(self, nb, label):
        return nb ** self.n, label


def power_of(nb, label, n):
    return nb ** n, label


def get_power_of(n):
    def power_of(nb, label):
        return nb ** n, label
    return power_of


def main():
    parser = argparse.ArgumentParser("Compare uses of class vs function for data augmentation")
    parser.add_argument("--nb_iterations", "--n", default=10_000_000, type=int, help="Number of iterations")
    # parser.add_argument("--images", "--i", action="store_true", help="Runs the test using images")
    args = parser.parse_args()

    start_time = time()
    power_of_transform = PowerOfDict(3)
    for i in range(args.nb_iterations):
        sample = {"nb": i, "label": i}
        power_of_transform(sample)
    print(f"Transform with dict took {1000*(time() - start_time):.2f}ms")

    start_time = time()
    power_of_transform = PowerOf(3)
    for i in range(args.nb_iterations):
        power_of_transform(i, i)
    print(f"Transform took {1000*(time() - start_time):.2f}ms")

    start_time = time()
    power_of_fn = get_power_of(3)
    for i in range(args.nb_iterations):
        power_of_fn(i, i)
    print(f"Function took {1000*(time() - start_time):.2f}ms")

    start_time = time()
    for i in range(args.nb_iterations):
        power_of(i, i, 3)
    print(f"Function took {1000*(time() -start_time):.2f}ms")


if __name__ == "__main__":
    main()
