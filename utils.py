from typing import Union


class AverageMeter(object):
    def __init__(self, init_c: float = 0., init_n: int = 0):
        self.c = init_c
        self.n = init_n
        self._recent = 0.  # the most recent value

    def reset(self):
        self.c, self.n, self._recent = 0., 0, 0.

    def set(self, c: Union[list, float], n: Union[list, float]):
        # assert check_argument_types()
        if isinstance(c, list):
            assert len(c) == n
            self.c += sum(c)
            self.n += n
            self._recent = sum(c) / n
        else:
            self.c += c
            self.n += n
            self._recent = c / n

    def update(self, *args, **kwargs):
        return self.set(*args, **kwargs)

    def get(self):
        return self.c / self.n if self.n != 0 else 0.

    @property
    def recent(self):
        return self._recent

    def __repr__(self):
        return self.get()
