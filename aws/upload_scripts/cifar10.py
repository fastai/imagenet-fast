from fastai import io
import tarfile
import shutil

from time import gmtime, strftime
from fastai.conv_learner import *
from fastai.models.cifar10.resnext import resnext29_8_64

PATH = "data/cifar10/"

# (AS) TODO: put this into the fastai library
def untar_file(file_path, save_path):
    if file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        obj = tarfile.open(file_path)
        obj.extractall(save_path)
        obj.close()
        os.remove(file_path)

def get_cifar():
    if os.path.exists('~/efs_mount_point/cifar10.tgz'):
        shutil.copy2('~/efs_mount_point/cifar10.tgz', 'data/cifar10.tgz')
    else:
        cifar_url = 'http://files.fast.ai/data/cifar10.tgz' # faster download
        # cifar_url = 'http://pjreddie.com/media/files/cifar.tgz'
        io.get_data(cifar_url, 'data/cifar10.tgz')

# Create cifar dataset
if not os.path.exists(PATH):
    os.makedirs(PATH,exist_ok=True)
    get_cifar()
    untar_file('data/cifar10.tgz', 'data/')

# Start training
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))

def get_data(sz,bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
    return ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)

bs=128


m = resnext29_8_64()
if torch.cuda.is_available():
    m = m.cuda()
    
bm = BasicModel(m, name='cifar10_rn29_8_64')

data = get_data(8,bs*4)

learn = ConvLearner(data, bm)
learn.unfreeze()

lr=1e-2; wd=5e-4

print('Skipping training for smaller instances')
# learn.fit(lr, 1)

learn.save('test_model')

if os.path.exists('data/models/'):
    print('Saving trained models')
    datestring = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    save_path = f'~/efs_mount_point/cifar10{datestring}/models/'
    os.makedirs(save_path,exist_ok=True)
    shutil.copy2('data/models/', save_path)