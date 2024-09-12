from typing import List, Tuple
from taskgraph import TaskGraph
import os
import math


class Bootstrapper:
    def __init__(self, log_q: int = 36, log_N: int = 16, L: int = 35) -> None:
        self.tg = TaskGraph(name="Bootstrapper", log_q=log_q, log_N=log_N, L=L)

        # parameters of (I)DFT
        self.k_dft = 5  # radix of dft is 2^{k_dft}
        self.fftIter = (self.tg.log_N - 1) // self.k_dft  # log_{2^k}{n}, n = N / 2
        self.k1_bsgs = 3  # k1/k2 for baby-step giant-step
        self.k2_bsgs = self.k_dft + 1 - self.k1_bsgs

        # https://github.com/snu-ccl/FHE-MP-CNN/blob/9b497a70a85a9eb0d73d06b2e0cd86f8db47c7bb/cnn_ckks/run/run_bootstrapping.cpp#L52
        self.deg = 59
        self.scale_factor = 2
        self.inverse_deg = 1
        self.num_double_formula = self.scale_factor

    def IDFT(self, m: int):
        """
        Evaluate the inverse DFT.
        m: index of the ciphertext to be evaluated.
        """
        t = m

        for s in range(self.fftIter):
            items_k2 = []
            for j in range(2**self.k2_bsgs):  # Eqn. (8)
                items_k1 = []
                for i in range(2**self.k1_bsgs):
                    rot_k1 = self.tg.HRotate(t, i * (2 ** (self.k_dft * s)))
                    it_k1 = self.tg.PMult(rot_k1, self.P_IDFT[s][i][j])
                    items_k1.append(it_k1)
                sum_k1 = self.tg.HAdd_BinTree(items_k1)
                it_k2 = self.tg.HRotate(
                    sum_k1, j * (2 ** (self.k1_bsgs + self.k_dft * s))
                )
                items_k2.append(it_k2)
            t = self.tg.HAdd_BinTree(items_k2)
        t = self.tg.Conjugate(t)  # TODO: is this correct?
        return t

    def DFT(self, m: int):
        """
        Homomorphically evaluate the DFT, refer to ARK.
        m: index of the ciphertext to be evaluated.
        """
        t = m

        for s in range(self.fftIter):
            items_k2 = []
            for j in range(2**self.k2_bsgs):  # Eqn. (8)
                items_k1 = []
                for i in range(2**self.k1_bsgs):
                    rot_k1 = self.tg.HRotate(t, i * (2 ** (self.k_dft * s)))
                    it_k1 = self.tg.PMult(rot_k1, self.P_DFT[s][i][j])
                    items_k1.append(it_k1)
                sum_k1 = self.tg.HAdd_BinTree(items_k1)
                it_k2 = self.tg.HRotate(
                    sum_k1, j * (2 ** (self.k1_bsgs + self.k_dft * s))
                )
                items_k2.append(it_k2)
            t = self.tg.HAdd_BinTree(items_k2)
        return t

    def CoeffToSlot(self, m: int) -> Tuple[int, int]:
        """
        Refer to:
            https://github.com/snu-ccl/FHE-MP-CNN/blob/9b497a70a85a9eb0d73d06b2e0cd86f8db47c7bb/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp#L2696
        """
        tmpct1 = self.IDFT(m=m)
        tmpplain = self.tg.add_input(
            is_ciph=False
        )  # TODO: same for all CoeffToSlot? if is, we should only add them into tg only once
        tmpct2 = self.tg.PMult(tmpct1, tmpplain)
        tempct3 = self.tg.Conjugate(idx_a=tmpct2)
        tempct4 = self.tg.Conjugate(idx_a=tmpct1)
        rtncipher1 = self.tg.HAdd(tmpct1, tempct4)
        rtncipher2 = self.tg.HAdd(tmpct2, tempct3)
        return rtncipher1, rtncipher2

    def SlotToCoeff(self, cipher1: int, cipher2: int) -> int:
        """
        Refer to:
            https://github.com/snu-ccl/FHE-MP-CNN/blob/9b497a70a85a9eb0d73d06b2e0cd86f8db47c7bb/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp#L2714
        """
        tmpplain = self.tg.add_input(
            is_ciph=False
        )  # TODO: same for all SlotToCoeff? If is, we should only add them into tg only once
        tmpct1 = self.tg.PMult(cipher2, tmpplain)
        tmpct3 = self.tg.HAdd_RedErr(cipher1, tmpct1)
        return self.DFT(m=tmpct3)

    def babycount(self, deg: int) -> Tuple[int, int]:
        """
        Refer to:
            https://github.com/snu-ccl/FHE-MP-CNN/blob/3646f9d91279858fae5ce4438794905b026e6fd1/cnn_ckks/common/func.cpp#L117
        """
        mink = 2
        d_over_k = deg / mink
        log2_d_over_k = math.ceil(math.log2(d_over_k))
        ceil_d_over_k = math.ceil(d_over_k)
        min_mul = log2_d_over_k + mink + ceil_d_over_k - 3
        minm = log2_d_over_k

        for i in range(3, math.floor(2 * math.sqrt(deg))):
            d_over_k = deg / i
            log2_d_over_k = math.ceil(math.log2(d_over_k))
            ceil_d_over_k = math.ceil(d_over_k)
            curr_mul = log2_d_over_k + i + ceil_d_over_k - 3

            if min_mul > curr_mul:
                mink = i
                min_mul = curr_mul
                minm = log2_d_over_k

        return mink, minm

    def oddbabycount(self, deg: int) -> Tuple[int, int]:
        mind = 0
        mineval = 100000
        for k in range(2, self.deg + 1, 2):
            m = 1
            while (1 << m) * k < self.deg:
                m += 1
            if math.ceil(math.log2(k)) + m == math.ceil(math.log2(self.deg)):
                if mineval > math.ceil(
                    (deg + 0.0) / (k + 0.0)
                ) + k / 2 - 5 + m + math.ceil(math.log(k)) + max(
                    3.0 - m, math.ceil((deg + 0.0) / (1.5 * (1 << (m - 1)) * k))
                ):
                    mineval = (
                        math.ceil((deg + 0.0) / (k + 0.0))
                        + k / 2
                        - 5
                        + m
                        + math.ceil(math.log(k))
                        + max(
                            3.0 - m, math.ceil((deg + 0.0) / (1.5 * (1 << (m - 1)) * k))
                        )
                    )
                    mink = k
                    minm = m
                    mind = math.ceil(math.log2(k)) + m
                elif mineval == math.ceil(
                    (deg + 0.0) / (k + 0.0)
                ) + k / 2 - 5 + m + math.ceil(math.log(k)) + max(
                    3.0 - m, math.ceil((deg + 0.0) / (1.5 * (1 << (m - 1)) * k))
                ):
                    if mind > math.ceil(math.log2(k)) + m:
                        mink = k
                        minm = m
                        mind = math.ceil(math.log2(k)) + m
        return mink, minm

    def generate_sin_cos_polynomial(self) -> None:
        """
        Refer to:
            https://github.com/snu-ccl/FHE-MP-CNN/blob/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp#L37
        """
        # TODO: `generate_optimal_poly` generates one plaintext or (deg + 1) plaintexts?
        self.sin_cos_polynomial = [None] * (self.deg + 1)
        for i in range(self.deg + 1):
            self.sin_cos_polynomial[i] = self.tg.add_input(
                name="scp-{}".format(i), is_ciph=False
            )  # scp is sin_cos_polynomial
        # `generate_poly_heap`
        self.scp_heap_k, self.scp_heap_m = self.babycount(deg=self.deg)
        # `generate_poly_heap_manual` does not need to generate the poly heap really
        # heaplen = (1 << (heap_m + 1)) - 1;
        self.scp_poly_heap = [
            [
                self.tg.add_input("scpph-{},{}".format(i, j), is_ciph=False)
                for j in range(self.deg + 1)
            ]
            for i in range((1 << (self.scp_heap_m + 1)) - 1)
        ]  # scpph is sin_cos_polynomial poly_heap

    def generate_inverse_sine_polynomial(self) -> None:
        """
        Refer to:
            https://github.com/snu-ccl/FHE-MP-CNN/blob/9b497a70a85a9eb0d73d06b2e0cd86f8db47c7bb/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp#L42
        """
        # TODO: `generate_optimal_poly` generates one plaintext or (deg + 1) plaintexts?
        self.inverse_sin_polynomial = [None] * (self.deg + 1)
        for i in range(self.deg + 1):
            self.inverse_sin_polynomial[i] = self.tg.add_input(
                name="isp-{}".format(i), is_ciph=False
            )  # isp is inverse_sin_polynomial
        if self.inverse_deg > 3:
            self.isp_heap_k, self.isp_heap_m = self.oddbabycount(deg=self.deg)
            self.isp_poly_heap = [
                self.tg.add_input("ispph-{}".format(i), is_ciph=False)
                for i in range((1 << (self.isp_heap_m + 1)) - 1)
            ]  # ispph is inverse_sin_polynomial poly_heap
        if self.inverse_deg == 1:
            self.scale_inverse_coeff = 0.5  # TODO:
            # ignore those operations related to `scale_inverse_coeff`
            self.isp_heap_k, self.isp_heap_m = self.oddbabycount(deg=self.deg)
            self.isp_poly_heap = [
                [
                    self.tg.add_input("ispph-{},{}".format(i, j), is_ciph=False)
                    for j in range(self.deg + 1)
                ]
                for i in range((1 << (self.isp_heap_m + 1)) - 1)
            ]  # ispph is inverse_sin_polynomial poly_heap

    def homomorphic_poly_evaluation(
        self, cipher: int, heap_k: int, heap_m: int, poly_heap: List[int]
    ) -> int:
        zero = 0  # TODO: is this a constant ?
        ## Polynomial.cpp#L321
        assert self.deg >= 3
        baby: List[int] = [None] * heap_k
        babybool: List[bool] = [False] * heap_k
        baby[1] = cipher
        babybool[1] = True

        i = 2
        while i < heap_k:
            baby[i] = self.tg.HMult(baby[i // 2], baby[i // 2])
            baby[i] = self.tg.HAdd(baby[i], baby[i])
            baby[i] = self.tg.PAdd(baby[i], self.neg_1)
            babybool[i] = True
            i = i * 2

        for i in range(1, heap_k):
            if not babybool[i]:
                lpow2 = 1 << math.floor(math.log2(i))
                res = i - lpow2
                diff = abs(lpow2 - res)
                baby[i] = self.tg.HMult_RedErr(
                    baby[lpow2], baby[res]
                )  # TODO: should baby[lpow2], baby[res] be the same level?
                baby[i] = self.tg.HAdd(baby[i], baby[i])
                baby[i] = self.tg.HSub_RedErr(baby[i], baby[diff])
                babybool[i] = True

        giant: List[int] = [None] * heap_m
        giantbool: List[bool] = [False] * heap_m
        giantbool[0] = True
        lpow2 = 1 << (math.ceil(math.log2(heap_k)) - 1)
        res = heap_k - lpow2
        diff = abs(lpow2) - res

        if res == 0:
            giant[0] = baby[lpow2]
        elif diff == 0:
            giant[0] = baby[lpow2]
            giant[0] = self.tg.HAdd(giant[0], giant[0])
            giant[0] = self.tg.PAdd(giant[0], self.neg_1)
        else:
            giant[0] = self.tg.HMult(baby[lpow2], baby[res])
            giant[0] = self.tg.HAdd(giant[0], giant[0])
            giant[0] = self.tg.HSub(giant[0], baby[diff])

        for i in range(1, heap_m):
            giantbool[i] = True
            giant[i] = self.tg.HMult(giant[i - 1], giant[i - 1])
            giant[i] = self.tg.HAdd(giant[i], giant[i])
            giant[i] = self.tg.PAdd(giant[i], self.neg_1)

        cipherheap = [None] * ((1 << (heap_m + 1)) - 1)
        cipherheapbool = [False] * ((1 << (heap_m + 1)) - 1)

        heapfirst = (1 << heap_m) - 1
        heaplast = (1 << (heap_m + 1)) - 1
        for i in range(heapfirst, heaplast):
            if poly_heap[i]:  # TODO: what is the type of poly_heap[i] ?
                cipherheapbool[i] = True
                cipherheap[i] = self.tg.PMult(baby[1], poly_heap[i][1])

                if not (abs(poly_heap[i][1]) <= zero):  # TODO:
                    cipherheap[i] = self.tg.PAdd(
                        cipherheap[i], poly_heap[i][0]
                    )  # TODO: what is poly_heap[i]->chebcoeff[0]

                for j in range(2, self.deg + 1):
                    if abs(poly_heap[i][j]) <= zero:
                        continue
                    if j < heap_k:
                        tmp = self.tg.PMult(baby[j], poly_heap[i][j])
                    else:
                        tmp = self.tg.PMult(giant[0], poly_heap[i][j])
                    cipherheap[i] = self.tg.HAdd_RedErr(tmp, cipherheap[i])
        depth = heap_m
        gindex = 0

        while depth != 0:
            depth -= 1
            heapfirst = (1 << depth) - 1
            heaplast = (1 << (depth + 1)) - 1
            for i in range(heapfirst, heaplast):
                if poly_heap[i]:
                    cipherheapbool[i] = True
                    if not cipherheapbool[2 * (i + 1) - 1]:
                        cipherheap[i] = cipherheap[2 * (i + 1)]
                    else:
                        cipherheap[i] = self.tg.HMult_RedErr(
                            cipherheap[2 * (i + 1) - 1], giant[gindex]
                        )
                        cipherheap[i] = self.tg.HAdd_RedErr(
                            cipherheap[i], cipherheap[2 * (i + 1)]
                        )
            gindex += 1
        rtn = cipherheap[0]
        return rtn

    def double_angle_formula(self, cipher: int):
        """
        Refer to:
            https://github.com/snu-ccl/FHE-MP-CNN/blob/9b497a70a85a9eb0d73d06b2e0cd86f8db47c7bb/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp#L21
        """
        cipher = self.tg.HMult(cipher, cipher)
        cipher = self.tg.HAdd(cipher, cipher)
        cipher = self.tg.PAdd(cipher, self.neg_1)
        return cipher

    def double_angle_formula_scaled(self, cipher: int, scale_coeff: float) -> int:
        """
        Refer to:
            https://github.com/snu-ccl/FHE-MP-CNN/blob/9b497a70a85a9eb0d73d06b2e0cd86f8db47c7bb/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp#L29
        """
        cipher = self.tg.HMult(cipher, cipher)
        cipher = self.tg.HAdd(cipher, cipher)
        scale_coeff = self.tg.add_input(
            name="sc-{:.4f}".format(-scale_coeff), is_ciph=False
        )  # sc is scale_coeff
        cipher = self.tg.PAdd(cipher, scale_coeff)
        return cipher

    def modular_reduction(self, cipher: int):
        """
        Refer to:
            https://github.com/snu-ccl/FHE-MP-CNN/blob/9b497a70a85a9eb0d73d06b2e0cd86f8db47c7bb/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp#L61
        """
        tmp1 = cipher
        tmp2 = self.homomorphic_poly_evaluation(
            cipher=tmp1,
            heap_k=self.scp_heap_k,
            heap_m=self.scp_heap_m,
            poly_heap=self.scp_poly_heap,
        )
        if self.inverse_deg == 1:
            curr_scale = self.scale_inverse_coeff
            for _ in range(1, self.num_double_formula):
                curr_scale = curr_scale * curr_scale
                tmp2 = self.double_angle_formula_scaled(tmp2, curr_scale)
            rtn = tmp2
        else:
            for _ in range(0, self.num_double_formula):
                tmp2 = self.double_angle_formula(cipher=tmp2)
            rtn = self.homomorphic_poly_evaluation(
                cipher=tmp2,
                heap_k=self.isp_heap_k,
                heap_m=self.isp_heap_m,
                poly_heap=self.isp_poly_heap,
            )
        return rtn

    def gen_tg(self) -> TaskGraph:
        """
        Generate the task graph of bootstrapping.
        Please execute this function only once!
        """
        self.neg_1 = self.tg.add_input(
            name="Const-{}".format(-1), is_ciph=False
        )  # plaintext encoding -1.0

        self.P_IDFT = []  # P[s][i][j] TODO: is this correct?
        self.P_DFT = []
        for _ in range(self.fftIter):
            self.P_IDFT.append([])
            self.P_DFT.append([])
            for __ in range(2**self.k1_bsgs):
                self.P_IDFT[-1].append([])
                self.P_DFT[-1].append([])
                for ___ in range(2**self.k2_bsgs):
                    self.P_IDFT[-1][-1].append(
                        self.tg.add_input(is_ciph=False)
                    )  # TODO: is P of IDFT same as that of DFT? if is, we should only add them into tg only once
                    self.P_DFT[-1][-1].append(self.tg.add_input(is_ciph=False))
        self.generate_sin_cos_polynomial()
        self.generate_inverse_sine_polynomial()

        m = self.tg.add_input(
            name="m", is_ciph=True
        )  # the ciphertext to be bootstrapped
        # refer to:
        # https://github.com/snu-ccl/FHE-MP-CNN/blob/9b497a70a85a9eb0d73d06b2e0cd86f8db47c7bb/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp#L3143
        m = self.tg.ModRaise(idx_a=m, l_new=self.tg.L)
        rtn1, rtn2 = self.CoeffToSlot(m)
        modrtn1 = self.modular_reduction(rtn1)
        modrtn2 = self.modular_reduction(rtn2)
        o = self.SlotToCoeff(modrtn1, modrtn2)
        self.tg.add_output(o)

        print("Generating PDF")
        self.tg.visualize(os.path.dirname(__file__))
        self.tg.check()
        self.tg.summarize()
        # generate_sin_cos_polynomial

        # generate_inverse_sine_polynomial

        return self.tg


if __name__ == "__main__":
    bs = Bootstrapper()
    bs.gen_tg()
