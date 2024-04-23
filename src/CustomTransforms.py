from torch import tensor


class FromHxWxC_To_CxHxW:
    def __call__(self, sample: tensor):
        return sample.permute(2, 0, 1)
