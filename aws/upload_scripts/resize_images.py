import os
from PIL import Image
import math
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import fire

def resize_img(fname, targ, path, new_path):
    dest = os.path.join(path,new_path,str(targ),fname)
    if os.path.exists(dest): return
    im = Image.open(os.path.join(path, fname)).convert('RGB')
    r,c = im.size
    ratio = targ/min(r,c)
    sz = (scale_to(r, ratio, targ), scale_to(c, ratio, targ))
    os.makedirs(os.path.split(dest)[0], exist_ok=True)
    im.resize(sz, Image.LINEAR).save(dest)

def resize_imgs(fnames, targ, path, new_path):
    if not os.path.exists(os.path.join(path,new_path,str(targ),fnames[0])):
        with ThreadPoolExecutor(16) as e:
            ims = e.map(lambda x: resize_img(x, targ, path, new_path), fnames)
            for x in tqdm(ims, total=len(fnames), leave=False): pass
    return os.path.join(path,new_path,str(targ))

def read_dir(path, folder):
    full_path = os.path.join(path, folder)
    fnames = glob(f"{full_path}/*.*")
    if any(fnames):
        return [os.path.relpath(f,path) for f in fnames]
    else:
        raise FileNotFoundError("{} folder doesn't exist or is empty".format(folder))

def read_dirs(path, folder):
    labels, filenames, all_labels = [], [], []
    full_path = os.path.join(path, folder)
    for label in sorted(os.listdir(full_path)):
        if label not in ('.ipynb_checkpoints','.DS_Store'):
            all_labels.append(label)
            for fname in os.listdir(os.path.join(full_path, label)):
                filenames.append(os.path.join(folder, label, fname))
                labels.append(label)
    return filenames, labels, all_labels

def scale_to(x, ratio, targ): return max(math.floor(x*ratio), targ)

def resize(targ, source_dir=None, resize_folder='resize'):
    if source_dir is None:
        source_dir = Path.home()/'data/imagenet'
    val_filenames, val_labels, val_all_labels = read_dirs(source_dir, 'val'); 
    print(f'Found {len(val_filenames)} validation images')

    train_filenames, train_labels, train_all_labels = read_dirs(source_dir, 'train'); len(train_filenames)
    print(f'Found {len(train_filenames)} training images')

    resize_imgs(train_filenames, targ, source_dir, resize_folder)
    resize_imgs(val_filenames, targ, source_dir, resize_folder)

if __name__ == '__main__':
  fire.Fire(resize)