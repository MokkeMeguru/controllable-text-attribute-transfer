import random


def merge(fnames):
    target = fnames[0]
    sources = fnames[1]
    result = []
    for fname, label in sources:
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                result.append(line.strip() + "\t" + str(label))
    random.shuffle(result)
    with open(target, "w", encoding="utf-8") as f:
        for line in result:
            f.write(line + "\n")


if __name__ == '__main__':
    train = (
        "./sentiment.train",
        [
            ["./sentiment.train.0", 0],
            ["./sentiment.train.1", 1]
        ])
    dev = (
        "./sentiment.dev",
        [
            ["./sentiment.dev.0", 0],
            ["./sentiment.dev.1", 1]
        ])
    test = (
        "./sentiment.test",
        [
            ["./sentiment.test.0", 0],
            ["./sentiment.test.1", 1]
        ])
    merge(train)
    merge(dev)
    merge(test)
