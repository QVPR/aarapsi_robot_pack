import cv2
import numpy as np

def grey2dToColourMap(matrix, colourmap=cv2.COLORMAP_JET, dims=None):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    matnorm = (((matrix - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
    if not (dims is None):
        matnorm = cv2.resize(matnorm, dims)
    mat_rgb = cv2.applyColorMap(matnorm, colourmap)
    return mat_rgb

def labelImage(img_in, textstring, org_in, colour):
# Write textstring at position org_in, with colour and black border on img_in

    # Black border:
    img_A = cv2.putText(img_in, textstring, org=org_in, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                color=(0,0,0), thickness=7, lineType=cv2.LINE_AA)
    # Colour inside:
    img_B = cv2.putText(img_A, textstring, org=org_in, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, \
                                color=colour, thickness=2, lineType=cv2.LINE_AA)
    return img_B

def convert_img_to_uint8(img, reshape=None, resize=None, dstack=True):
    if not type(img.flatten()[0]) == np.uint8:
        _min      = np.min(img)
        _max      = np.max(img)
        _img_norm = (img - _min) / (_max - _min)
        img       = np.array(_img_norm * 255, dtype=np.uint8)
    if not (reshape is None):
        img = np.reshape(img, reshape)
    if not (resize is None):
        img = cv2.resize(img, resize, interpolation = cv2.INTER_AREA)
    if dstack: return np.dstack((img,)*3)
    return img

def makeImage(query_raw, match_raw, img_dims, icon_dict):
# Produce image to be published via ROS that has a side-by-side style of match (left) and query (right)
# Query image comes in via cv2 variable query_raw
# Match image comes in from ref_dict

    match_img   = convert_img_to_uint8(match_raw, reshape=img_dims, resize=(500,500), dstack=(not len(match_raw.shape) == 3))
    query_img   = convert_img_to_uint8(query_raw, reshape=img_dims, resize=(500,500), dstack=(not len(query_raw.shape) == 3))

    icon_to_use = icon_dict['icon']
    icon_size   = icon_dict['size']
    icon_dist   = icon_dict['dist']
    
    match_img_lab = labelImage(match_img, "Reference", (20,40), (100,255,100))
    query_img_lab = labelImage(query_img, "Query", (20,40), (100,255,100))

    if icon_size > 0:
        # Add Icon:
        img_slice = query_img_lab[-1 - icon_size - icon_dist:-1 - icon_dist, -1 - icon_size - icon_dist:-1 - icon_dist, :]
        # https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
        icon_mask_inv = cv2.inRange(icon_to_use, (50,50,50), (255,255,255)) # get border (white)
        icon_mask = 255 - icon_mask_inv # get shape
        icon_mask_stack_inv = cv2.merge([icon_mask_inv, icon_mask_inv, icon_mask_inv]) / 255 # stack into rgb layers, binary image
        icon_mask_stack = cv2.merge([icon_mask, icon_mask, icon_mask]) / 255 # stack into rgb layers, binary image
        opacity_icon = 0.8 # 80%
        # create new slice with appropriate layering
        img_slice = (icon_mask_stack_inv * img_slice) + \
                    (icon_mask_stack * icon_to_use) * (opacity_icon) + \
                    (icon_mask_stack * img_slice) * (1-opacity_icon)
        query_img_lab[-1 - icon_size - icon_dist:-1 - icon_dist, -1 - icon_size - icon_dist:-1 - icon_dist, :] = img_slice

    return np.concatenate((match_img_lab, query_img_lab), axis=1)