import networkx as nx
import pydot
import os
from typing import Union, List


class TaskGraph:
    def __init__(self, name: str, log_q: int, log_N: int, L: int):
        self.name = name  # name of the task graph
        self.g = nx.MultiDiGraph()
        self.log_q = log_q  # bit size of modulus
        self.log_N = log_N  # number of slots
        self.L = L  # maximum number of levels

    def get_is_ciph(self, idx_n: int) -> bool:
        """
        Return True if the output of the producer idx_n is a ciphertext.
        """
        is_op = self.g.nodes[idx_n]["is_op"]
        if is_op:  # the output of a operation is always a ciphertext
            return True
        else:  # is input
            assert self.g.nodes[idx_n]["is_input"]
            return self.g.nodes[idx_n]["is_ciph"]

    def add_input(self, is_ciph: bool = True, l: Union[int, None] = None):
        """
        Add a ciphertext/plaintext of plaintext input node to the task graph.
        l: level of ciphertext
        """
        if l is None:
            l = self.L
        idx_n = len(self.g)
        self.g.add_node(
            idx_n,
            is_op=False,
            is_input=True,
            is_ciph=is_ciph,
            l=self.L if is_ciph else -1,
        )
        return idx_n

    def add_output(self, idx_prod: int):
        """
        Add a output node to the task graph.
        idx_prod: index of the kernel node that produces the output.
        """
        idx_n = len(self.g)
        self.g.add_node(idx_n, is_op=False, is_input=False, is_ciph=True)
        self.g.add_edge(idx_prod, idx_n, map_od=0, l=self.g.nodes[idx_prod]["l"])
        return idx_n

    def HAdd(self, idx_a: int, idx_b: int):
        """
        idx_a/idx_b: index of the left/right operand producer kernel.
        """
        assert self.get_is_ciph(idx_a) and self.get_is_ciph(idx_b)
        idx_n = len(self.g)
        l_a, l_b = self.g.nodes[idx_a]["l"], self.g.nodes[idx_b]["l"]
        assert l_a == l_b  # inputs should have the same level
        self.g.add_node(idx_n, is_op=True, op="HAdd", l=l_a)
        self.g.add_edge(idx_a, idx_n, map_od=0, l=l_a)  # map_od is 0 for left operand
        self.g.add_edge(idx_b, idx_n, map_od=1, l=l_a)  # map_od is 1 for right operand
        return idx_n

    def HSub(self, idx_a, idx_b):
        assert self.get_is_ciph(idx_a) and self.get_is_ciph(idx_b)
        idx_n = len(self.g)
        l_a, l_b = self.g.nodes[idx_a]["l"], self.g.nodes[idx_b]["l"]
        assert l_a == l_b
        self.g.add_node(idx_n, is_op=True, op="HSub", l=l_a)
        self.g.add_edge(idx_a, idx_n, map_od=0, l=l_a)
        self.g.add_edge(idx_b, idx_n, map_od=1, l=self.get_l(idx_b))
        return idx_n

    def HMult(self, idx_a, idx_b):
        assert self.get_is_ciph(idx_a) and self.get_is_ciph(idx_b)
        idx_n = len(self.g)
        l_a, l_b = self.g.nodes[idx_a]["l"], self.g.nodes[idx_b]["l"]
        assert l_a == l_b
        self.g.add_node(idx_n, is_op=True, op="HMult", l=l_a - 1)
        self.g.add_edge(idx_a, idx_n, map_od=0, l=l_a)
        self.g.add_edge(idx_b, idx_n, map_od=1, l=l_b)
        return idx_n

    def PMult(self, idx_a, idx_b):
        """
        Ciphertext-Plaintext multiplication.
        """
        assert self.get_is_ciph(idx_a) != self.get_is_ciph(idx_b)
        idx_n = len(self.g)
        l_a, l_b = self.g.nodes[idx_a]["l"], self.g.nodes[idx_b]["l"]
        l = max(l_a, l_b)
        self.g.add_node(idx_n, is_op=True, op="PMult", l=l - 1)
        self.g.add_edge(idx_a, idx_n, map_od=0, l=l)
        self.g.add_edge(idx_b, idx_n, map_od=1, l=l)
        return idx_n

    def HRotate(self, idx_a: int, pos_rot: int):
        assert self.get_is_ciph(idx_a)
        idx_n = len(self.g)
        l = self.g.nodes[idx_a]["l"]
        self.g.add_node(idx_n, is_op=True, op="HRotate", l=l, pos_rot=pos_rot)
        self.g.add_edge(idx_a, idx_n, map_od=0, l=l)
        return idx_n

    def ModRaise(self, idx_a: int):
        assert self.get_is_ciph(idx_a)
        idx_n = len(self.g)
        l = self.g.nodes[idx_a]["l"]
        self.g.add_node(idx_n, is_op=True, op="ModRaise", l=l + self.K + 1)  # TODO:
        self.g.add_edge(idx_a, idx_n, map_od=0, l=l + self.K + 1)
        return idx_n

    def Conjugate(self, idx_a):
        assert self.get_is_ciph(idx_a)
        idx_n = len(self.g)
        l = self.g.nodes[idx_a]["l"]
        self.g.add_node(idx_n, is_op=True, op="Conjugate", l=l)
        self.g.add_edge(idx_a, idx_n, map_od=0, l=l)
        return idx_n
    
    def HAdd_BinTree(self, indices: List[int]):
        """
        Generate a binary tree of HAdd operations.
        indices: indices of the producers.
        """
        if len(indices) == 1:
            return indices[0]
        else:
            mid = len(indices) // 2
            idx_a = self.HAdd_BinTree(indices[:mid])
            idx_b = self.HAdd_BinTree(indices[mid:])
            return self.HAdd(idx_a, idx_b)

    def check(self):
        """
        Check if the task graph is valid.
        """
        # the graph should be directed acyclic
        if not nx.is_directed_acyclic_graph(self.g):
            raise ValueError("The task graph is not a directed acyclic graph.")
        # check the level of each homomorphic operation
        for idx_n in self.g.nodes():
            prevs = list(self.g.predecessors(idx_n))
            if len(prevs) > 1:
                if len(prevs) != 2:
                    raise ValueError(
                        "Too many predecessors for a homomorphic operation {}.".format(
                            idx_n
                        )
                    )
                prev_a, prev_b = prevs
                if self.g[prev_a][idx_n][0]["l"] != self.g[prev_b][idx_n][0]["l"]:
                    raise ValueError(
                        "The levels of the two operands of a homomorphic operation {} are different.".format(
                            idx_n
                        )
                    )
        for u, v, attr_e in self.g.edges(data=True):
            if attr_e["l"] < 0:
                raise ValueError(
                    "The level of the edge {}->{} is negative.".format(u, v)
                )
    
    def summarize(self):
        """
        Print statistical data.
        """
        print("Number of operations: {}".format())
        print("Number of homomorphic rotations: {}".format())

    def visualize(self, dir_pdf: str):
        """
        dir_pdf: directory to save the pdf file.
        """
        g_pdot = pydot.Dot(name=self.name, rankdir="TB")  # TB is top to bottom
        g_pdot.set_node_defaults()
        g_pdot.set_edge_defaults()
        for idx_n, attr_n in self.g.nodes(data=True):
            if attr_n["is_op"]:
                n_pdot = pydot.Node(
                    str(idx_n), label="{}-{}".format(idx_n, attr_n["op"])
                )
            else:
                n_pdot = pydot.Node(
                    str(idx_n),
                    label="{}-{}".format(idx_n, "In" if attr_n["is_input"] else "Out"),
                )
            g_pdot.add_node(n_pdot)
        for u, v, attr_e in self.g.edges(data=True):
            le_pdot = "{},l={}".format(attr_e["map_od"], attr_e["l"])
            e_pdot = pydot.Edge(str(u), str(v), label=le_pdot)
            g_pdot.add_edge(e_pdot)
        g_pdot.write_pdf(os.path.join(dir_pdf, self.name + ".pdf"))


def example():
    """
    B = A * A
    D = B + C
    E = D + D
    """
    tg = TaskGraph(name="example", log_q=36, log_N=16, L=34)
    A = tg.add_input(is_ciph=True)
    C = tg.add_input(is_ciph=False)
    B = tg.HMult(A, A)
    D = tg.PMult(C, B)
    E = tg.HAdd(D, D)
    tg.add_output(E)
    tg.visualize(os.path.dirname(__file__))
    tg.check()


if __name__ == "__main__":
    example()
