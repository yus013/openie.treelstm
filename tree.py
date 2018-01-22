# tree object from stanfordnlp/treelstm
class Tree(object):
    def __init__(self):
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for ch in children:
            count += ch.size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for ch in self.children:
                child_depth = ch.depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth
