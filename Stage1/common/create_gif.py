import os
import imageio
import natsort  # for actually sort file

def create_gif(png_dir='../Media/png', fps=24):
    images = []
    for file_name in natsort.natsorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            try:
                images.append(imageio.imread(file_path))
            except ValueError:
                pass
    imageio.mimsave(os.path.join(png_dir, 'movie.gif'), images, fps=fps)
    print('Gif create.')

create_gif('C:/Users\Anton\PycharmProjects\Project-bI\Media\png')