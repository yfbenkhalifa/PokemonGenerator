from torch.utils.data.dataset import Dataset
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
def visualize(dataset: Dataset, index: int, category: str = 'image'):
    image_tensor = dataset[category][index]

    # Convert the tensor to a numpy array
    image_data = image_tensor.numpy()

    # Transpose the dimensions from CxHxW to HxWxC
    image_data = np.transpose(image_data, (1, 2, 0))

    # Display the image
    plt.imshow(image_data)
    plt.show()