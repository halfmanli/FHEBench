from typing import List
from taskgraph import TaskGraph
import os


def CoeffToSlot(tg: TaskGraph, m: int):
    """
    Evaluate the inverse DFT.
    m: index of the ciphertext to be evaluated.
    """
    P = []  # P_{s_i} is the
    for _ in range():
        P.append(tg.add_input(is_ciph=False))

    radix = 5
    log_N = tg.log_N
    fftIter = (log_N - 1) // radix  # log_{2^k}{n}, n = N / 2
    for s in range(fftIter):
        for i_k2 in range():
            tg.HAdd_BinTree()


def SlotToCoeff(tg: TaskGraph, m: int):
    """
    Homomorphically evaluate the DFT, refer to ARK.
    m: index of the ciphertext to be evaluated.
    """
    k = 5  # radix of FFT
    log_N = tg.log_N
    fftIter = (log_N - 1) // k  # log_{2^k}{n}, n = N / 2
    k1 = 3
    k2 = k + 1 - k1
    t = m

    P = []  # P[s][i][j]
    for s in range(fftIter):
        P.append([])
        for i in range(2**k1):
            P[-1].append([])
            for j in range(2**k2):
                P[-1][-1].append(tg.add_input(is_ciph=False))

    for s in range(fftIter):
        items_k2 = []
        for j in range(2**k2):  # Eqn. (8)
            items_k1 = []
            for i in range(2**k1):
                rot_k1 = tg.HRotate(t, i * 2 ** (k * s))
                it_k1 = tg.PMult(rot_k1, P[s][i][j])
                items_k1.append(it_k1)
            sum_k1 = tg.HAdd_BinTree(items_k1)
            it_k2 = tg.HRotate(sum_k1, j * (2 ** (k1 + k * s)))
            items_k2.append(it_k2)
        t = tg.HAdd_BinTree(items_k2)
    return t


def Bootstrap():
    tg = TaskGraph(name="Bootstrap", log_q=36, log_N=16, L=35)
    m = tg.add_input(is_ciph=True)  # the ciphertext need to be bootstrapped
    SlotToCoeff(tg, m)
    tg.visualize(os.path.dirname(__file__))
    tg.check()
    exit(0)
    fftIter = 5
    # CoeffToSlot
    for s in range(fftIter):
        ...

    # modular reduction

    # SlotToCoeff


if __name__ == "__main__":
    Bootstrap()
