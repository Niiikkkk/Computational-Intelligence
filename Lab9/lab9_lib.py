# Copyright © 2023 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free for personal or classroom use; see 'LICENSE.md' for details.

from abc import abstractmethod


class AbstractProblem:
    def __init__(self):
        self._calls = 0

    @property
    @abstractmethod
    def x(self):
        pass

    @property
    def calls(self):
        return self._calls

    @staticmethod
    def onemax(genome): #returns the number of 1 in the genome
        return sum(bool(g) for g in genome)

    def __call__(self, genome):
        self._calls += 1
        #genome[s::self.x] -> from s to end skipping self.x
        #onemax(...) -> count number of 1
        #so finsesses will be an array of self.x elements sorted
        fitnesses = sorted((AbstractProblem.onemax(genome[s :: self.x]) for s in range(self.x)), reverse=True)
        #sum(f for f in fitnesses if f == fitnesses[0]) count the number of elements in fintsesses that
        # has value equal to the maximum that is in the 0 position
        val = sum(f for f in fitnesses if f == fitnesses[0]) - sum(
            f * (0.1 ** (k + 1)) for k, f in enumerate(f for f in fitnesses if f < fitnesses[0])
        )
        return val / len(genome)


def make_problem(a):
    class Problem(AbstractProblem):
        @property
        @abstractmethod
        def x(self):
            return a
    return Problem()
