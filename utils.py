import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
from IPython import display
from tqdm import tqdm
from PIL import Image
import cv2
from time import time
import pylab

def _vgg(depth, padding_method="reflect"):
  if depth == 19:
    vgg = tf.keras.applications.VGG19(include_top=False)
  elif depth == 16:
    vgg = tf.keras.applications.VGG16(include_top=False)
  else:
    raise ValueError(f"depth can only be 16 or 19, got {depth}")
    
  Input = tf.keras.layers.Input((None,None,3))  
  x = Input
  for l in vgg.layers[1:]:
    if isinstance(l, tf.keras.layers.Conv2D):
      config = l.get_config()
      config["padding"] = "valid"
      nl = l.from_config(config)
      x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], mode="SYMMETRIC", name=l.name+"_pad")
      x = nl(x)
      for wn, w in zip(nl.weights, l.weights):
        wn.assign(w)
    else:
      config = l.get_config()
      nl = l.from_config(config)
      x = nl(x)
  return tf.keras.Model(Input, x)


def VGG(depth=19,
        ret_layers=[(2,1), (1,1), (2,1), (3,1), (4,1)],
        preprocess=True,
        padding_mode="reflect"):
  
  assert len(ret_layers) > 0
  vgg = _vgg(depth, padding_mode)
  if isinstance(ret_layers[0], tuple):
    ret_layers = [f"block{i}_conv{j}" for i,j in ret_layers]
  preprocessor = tf.keras.applications.vgg19.preprocess_input
    
    
  ret_model  = tf.keras.Model(
    inputs = vgg.inputs,
    outputs=[vgg.get_layer(name).output for name in ret_layers],
    name=f"vgg{depth}"
  )
    
  if preprocess:
    Input = tf.keras.Input((None, None, 3))
    x = preprocessor(Input)
    outputs = ret_model(x)
    ret_model = tf.keras.Model(
      inputs=[Input],
      outputs=outputs,
      name=f"vgg{depth}_with_preprocess"
    )
  
  dummy_dict = {
    f"vgg{depth}":vgg,
    "preprocessor" : preprocessor,
  }
    
  return ret_model, dummy_dict

def load_image(path, base=None):
  img = image.img_to_array(image.load_img(path))
  if base:
    img = img[:img.shape[0]//base*base, :img.shape[1]//base*base, :]
  return tf.constant(img)

def resize(img, target=None, base=None, mul=None):
  flag = 0
  for _ in [target, base, mul]:
    if _ is not None: flag += 1
  
  if not flag:
    raise ValueError("At least one of the arguments (target, base, mul) should have value")
  if flag > 1:
    raise NotImplementedError("Not implement with both (target, base, mul) hava value")
  
  if target is not None:
    img = tf.image.resize(img, target)
    return img 
  if base is not None:
    target = list(map(lambda x:x//base*base, img.shape[:2]))
  if mul is not None:
    target = list(map(lambda x:int(x*mul), img.shape[:2]))
  return resize(img, target)
    

def gen_gif(array, path, frames_cap = 100, show=True):
  frames_cap = min(frames_cap, len(array))
  imgs = [Image.fromarray((img*255).astype(np.uint8)) for img in array]
  if frames_cap > 1:
    indexes = np.linspace(1,len(imgs)-1,frames_cap-1).astype(np.int)
  else:
    indexes = range(1,len(imgs))

  imgs[0].save(path, save_all=True,
               append_images=[imgs[i] for i in indexes],
               duration=10, loop=1000)

  print(f"save at {path}")
  if show:
    return display.HTML('<img src="{}">'.format(path))
  
def blur(image, sigma=1):
  ksize = max(int((sigma*4)//2*2+1), 3)
  return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def special_error_function(x,y):
  return tf.reduce_mean(tf.abs(x**2-y**2))

def error_function(x,y):
  return tf.reduce_mean((x-y)**2)

def content_loss(tensors1, tensors2, error_function=error_function):
  return tf.add_n([error_function(t1, t2) for t1, t2 in zip(tensors1,tensors2)])/len(tensors1)

def style_loss(tensors1, tensors2, error_function=special_error_function):
  return tf.add_n([error_function(
    gram_matrix(t1),
    gram_matrix(t2)
  ) for t1, t2 in zip(tensors1, tensors2)])/len(tensors1)

def gram_matrix(tensor):
  shape = tensor.shape
  assert len(shape) == 4, "We do not implement for tensor shape is not euqal to 4"
  denumerator = shape[1]*shape[2]
  return tf.einsum('bijc,bijk->bck', tensor, tensor)/denumerator 

def total_variation(tensor):
  dy_abs = (tensor[:, 1:, :, :]-tensor[:, :-1, :, :])**2
  dx_abs = (tensor[:, :, 1:, :]-tensor[:, :, :-1, :])**2
  return tf.reduce_mean(dy_abs)+tf.reduce_mean(dx_abs)

def pyrDown(src, depth):
  for _ in range(depth):
    rows, cols = src.shape[:2]
    src = cv2.pyrDown(src, dstsize=(cols // 2, rows // 2))
  return src

def plot_images(array, grid_layout=(3,3), str_format="iter : {}", title=True):
  steps = grid_layout[0]*grid_layout[1]
  for i, ith_step in enumerate(np.linspace(0,len(array)-1,num=steps)):
    ith_step = ith_step.astype(np.int)
    plt.subplot(*grid_layout,i+1)
    if title:
      plt.title(str_format.format(ith_step))
    plt.axis("off")
    plt.imshow(array[ith_step])
  plt.show()