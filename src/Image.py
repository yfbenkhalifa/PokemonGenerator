@staticmethod
def downscale(width: int, height: int, factor: int) -> Tuple[int, int]:
    return width // factor, height // factor