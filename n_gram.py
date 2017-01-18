class NGram:
    def __init__(self, seq):
        self.n = len(seq)
        self.seq = seq

    def __hash__(self):
        result = 17
        result = 31 * result + self.n
        result = 31 * result + hash(self.seq)
        return result

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.seq == other.seq and self.n == other.n

        return False

    def __ne__(self, other):
        return not self.__eq__(other)
