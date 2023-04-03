#!/usr/bin/env python3

# https://github.com/spaceml-org/Missing-Pixel-Filler/blob/main/swathfiller_demo.ipynb

import math
import random 
import numpy as np
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.backends.cudnn.benchmark = True

# input: img (np array)
# output: arr (np array with swath filled by random RGB)
def fill_swath_with_random_rgb(img, color=[0,0,0]):
  arr = img.copy()
  x, y, z = np.where(arr==color)
  for i in range(len(x)):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    arr[x[i]][y[i]]=color
  return arr

# input: x_arr (non-swath x coords), y_arr (non-swath y coords)
# output: a random non-swath pixel
def get_random_pixel_from_image(x_arr, y_arr):
  index = random.randint(0, len(x_arr)-1)
  return x_arr[index], y_arr[index]

# input: img (np array)
# output: img (np array with random other pixels from image)
def fill_swath_with_random_pixel_from_image_new(img, color=[0,0,0]):
  img = img.copy()
  (x_non_swath, y_non_swath, z_non_swath) = np.where(img != color)
  (x_swath, y_swath, z_swath) = np.where(img == color)
  for i in range(len(x_swath)):
    x_pixel, y_pixel = get_random_pixel_from_image(x_non_swath, y_non_swath)
    img[x_swath[i]][y_swath[i]] = img[x_pixel][y_pixel]

  return img

# Dynamically tries finding non empty points in neighbourhood
# When it fails few times, it increases neighbourhood size automatically
def get_neighboring_pixel(img, x, y):
  x_rand, y_rand = 0,0

  max_num_tries = 30
  max_tries_per_neighbourhood = 3
  neighbourhood_size_increment = 10
  current_window_size = 10
  total_tries = 0
  for _ in range(math.ceil(max_num_tries/max_tries_per_neighbourhood)):
    for _ in range(max_tries_per_neighbourhood):
      min_x = max(0, x-current_window_size)
      max_x = min(224, x+current_window_size)
      min_y = max(0, y-current_window_size)
      max_y = min(224, y+current_window_size)

      #print((min_x, max_x-1, min_y, max_y-1))
      x_rand = random.randint(min_x, max_x-1)
      y_rand = random.randint(min_y, max_y-1)
      total_tries += 1
      if not(img[x_rand][y_rand][0]==0 and img[x_rand][y_rand][1]==0 and img[x_rand][y_rand][2]==0):
        return x_rand, y_rand
    current_window_size += neighbourhood_size_increment

  return x_rand, y_rand

# Owen's corrections ...
# Dynamically tries finding non empty points in neighbourhood
# When it fails few times, it increases neighbourhood size automatically
def get_neighboring_pixel_random_fix(img, x, y, img_w, img_h, startsize=10, stepsize=10):
  x_rand, y_rand = 0,0

  max_tries = 30
  max_tries_per_step = 3

  for _ in range(math.ceil(max_tries/max_tries_per_step)): # over number of neighbourhoods
    for _ in range(max_tries_per_step): # over each neighbourhood attempt
      min_x = max(0,        x-startsize) # don't go below 0
      max_x = min(img_w-1,  x+startsize) # don't go above window size
      min_y = max(0,        y-startsize)
      max_y = min(img_h-1,  y+startsize)

      x_rand = random.randint(min_x, max_x)
      y_rand = random.randint(min_y, max_y)
      
      if all(img[x_rand][y_rand] != [0,0,0]):
        return x_rand, y_rand
    startsize += stepsize

  return x_rand, y_rand


# input: img (np array)
# output: img (np array with corrected pixels using nearest neighbours)
def fill_swath_with_neighboring_pixel(img, startsize=10, stepsize=10):
  img_with_neighbor_filled = img.copy()
  (x_swath, y_swath, z_swath) = np.where(img == [0, 0, 0])
  img_w, img_h, _ = img.shape
  for i in range(len(x_swath)):
    x_rand, y_rand = get_neighboring_pixel_random_fix(img, x_swath[i], y_swath[i], img_w, img_h, startsize, stepsize)
    img_with_neighbor_filled[x_swath[i]][y_swath[i]] = img[x_rand][y_rand]
  return img_with_neighbor_filled

# input: img (np array)
# output: img (np array with corrected pixels using nearest neighbours)
def fill_swath_fast(img_in):
  img = img_in.copy()
  img_t = torch.from_numpy(img).to(device)

  img_h_t, img_w_t, _ = img_t.shape

  flatind_t = torch.nonzero(torch.sum(img_t,2).flatten()==0).flatten()
  y_swath_t = torch.floor(flatind_t / img_w_t).to(torch.long)
  x_swath_t = flatind_t % img_w_t

  u_t       = torch.arange(img_w_t).repeat(img_h_t).to(device)
  v_t       = torch.transpose(torch.arange(img_h_t).repeat(img_w_t,1),0,1).flatten().to(device)

  va_t  = torch.sub(u_t[:, None], x_swath_t)
  vb_t  = torch.sub(v_t[:, None], y_swath_t)
  vas_t = torch.pow(va_t, 2)
  vbs_t = torch.pow(vb_t, 2)
  vec_t = torch.add(vas_t, vbs_t)

  vec_t[flatind_t] = torch.max(vec_t, dim=0)[0] # return 0th list (max values only, we don't care for indices)

  closest_points_t = torch.argmin(vec_t, dim=0)
  x_indices_t = closest_points_t % img_w_t
  y_indices_t = torch.floor(closest_points_t / img_w_t).to(torch.long)
  img_t[y_swath_t, x_swath_t] = img_t[y_indices_t, x_indices_t]
  
  img_out = img_t.cpu().numpy()

  return img_out