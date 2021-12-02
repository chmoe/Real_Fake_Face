# %%
from skimage.segmentation import slic, mark_boundaries
from skimage import io, color
from tqdm import tqdm

# %%
def path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
def get_image_file_list(path):
    image_filenames = [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]
    return image_filenames

train_file_path = "./result/"
fake_image_filenames = get_image_file_list(train_file_path + "fake")
real_image_filenames = get_image_file_list(train_file_path + "real")

def save_img(name, arr, j):
    path = './SLIC_result/' + name + '/{}.png'.format(j)
    io.imsave(path, arr)

def process(img, segment, compactness):
    rgb = io.imread(img)
    segments = slic(rgb, n_segments=segment, compactness=compactness,)
    superpixels = color.label2rgb(segments, rgb, kind='avg')
    return superpixels

def run(list_name, name):
    file_count = len(list_name)
    start_count = len(get_image_file_list(path_exist('./SLIC_result/' + name)))
    for i in tqdm(range(start_count, file_count)):
        # print(name + "Image: Processing({}/{})".format(i + 1, file_count))
        arr = process(list_name[i], 40, 10)
        save_img(name, arr, i)
        
if __name__ == '__main__':
    run(fake_image_filenames, 'fake')
    run(real_image_filenames, 'real')
# %%

# 参考

# https://stackoverflow.com/questions/41578473/how-to-calculate-average-color-of-a-superpixel-in-scikit-image