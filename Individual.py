import cv2
import numpy as np
import matplotlib.pyplot as plt
from colour import delta_E


class Individual:
    def __init__(self, path_to_img: str = '', size: tuple = ()) -> None:
        if len(path_to_img) == 0:
            self.img = np.zeros(size, dtype=np.float32)
            self.size = size
            self.random_initialize()
        else:
            self.img = cv2.imread(path_to_img)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.img = np.array(self.img, dtype=np.float32) / 255
            self.size = self.img.shape
        
        self.img_lab = cv2.cvtColor(self.img, cv2.COLOR_RGB2Lab)

    def add_random_block(self, color: np.ndarray) -> None:
        min_len, max_len = int(0.01*self.size[0]), int(0.1*self.size[1])
        block_width = np.random.randint(min_len, max_len)
        start_width = np.random.randint(self.size[0])
        end_width = start_width
        if start_width + block_width >= self.size[0]:
            end_width -= block_width
        elif start_width - block_width < 0:
            end_width += block_width
        else:
            end_width += np.random.choice([-1, 1])*block_width
        if start_width > end_width:
            start_width, end_width = end_width, start_width
            
        block_height = np.random.randint(min_len, max_len)
        start_height = np.random.randint(self.size[1])
        end_height = start_height
        if start_height + block_height >= self.size[1]:
            end_height -= block_height
        elif start_height - block_height < 0:
            end_height += block_height
        else:
            end_height += np.random.choice([-1, 1])*block_height
        if start_height > end_height:
            start_height, end_height = end_height, start_height
        
        self.img[start_width:end_width, start_height:end_height] = color

    def random_initialize(self) -> None:
        background_color = np.random.random(3)
        self.img[:, :] = background_color
        random_color = np.random.random(3)
        for _ in range(np.random.randint(20, 50)):
            self.add_random_block(random_color)
    
    def fitness(self, target_img: np.ndarray) -> np.float32:
        target_img_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2Lab)
        delta_e = delta_E(self.img_lab, target_img_lab)
        delta_e = np.mean(delta_e)
        
        return delta_e

    def mutate(self, rate: float, max_blocks: int) -> None:
        blocks = range(max_blocks+1)
        p = [rate**i for i in range(1, max_blocks+1)]
        p.insert(0, 1-rate)
        p = np.array(p, dtype=np.float32)
        p = p / np.sum(p)
        num_blocks = np.random.choice(blocks, p=p)
        
        for _ in range(num_blocks):
            random_color = np.random.random(3)
            self.add_random_block(random_color)

    def show(self) -> None:
        plt.imshow(self.img)
        plt.show()