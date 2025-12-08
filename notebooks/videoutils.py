from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib as mpl
import pickle

mpl.rcParams['animation.embed_limit'] = 100  


def make_animation(frames_list, interval=40):
    images = []
    
    for p in frames_list:
        with Image.open(p) as img:
            images.append(img.copy())

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")

    def anim(i):
        ax.clear()
        ax.axis("off")
        ax.imshow(images[i])
        
    ani = FuncAnimation(fig, anim, frames=len(images), interval=interval, repeat=False)

    plt.close(fig)
    return HTML(ani.to_jshtml())