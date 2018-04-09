import imageio

def create_gif(gif_name, count):
    frames = []
    for i in range(count):
        frames.append(imageio.imread(str(i)+".png"))
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.8)



if __name__ == "__main__":
    create_gif("1.gif",20)