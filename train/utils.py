import queue


def equalIgnoreCase(a, b):
    return a.lower() == b.lower()


def insertQ(q, x):
    try:
        q.put_nowait(x)
    except queue.Full as e:
        print(e)


