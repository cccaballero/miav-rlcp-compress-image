import pandas as pd

import numpy as np
from PIL import Image, ImageCms

# Open image and discard alpha channel which makes wheel round rather than square
im = Image.open('image2.png').convert('RGB')

if im.mode != "RGB":
  im = im.convert("RGB")


pix_list_rgb = list(im.getdata())

srgb_profile = ImageCms.createProfile("sRGB")
lab_profile  = ImageCms.createProfile("LAB")
rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
lab_im = ImageCms.applyTransform(im, rgb2lab_transform)
pix_list2 = list(lab_im.getdata())

pix_sample2 = pix_list2
learn_matrix2 = pd.DataFrame(pix_list2, columns=["R1", "R2", "R3"])

import colour

def compare_objects(o1, o2):
    return colour.difference.delta_E_CIE1994(o1, o2)
    # return 1 - ((colour.delta_E(o1, o2)*100/206.8043) / 100)

from multiprocessing import Process, Array
from itertools import combinations_with_replacement, islice, tee, chain

import ctypes as c



def grouper(size, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, size))
        if not chunk:
            break
        yield chunk


beta = 5
groups = []

chunk = 0
chunk_size = 5000
n_processes = 8
for pix_chunk in grouper(chunk_size, pix_sample2):
    n = len(pix_chunk)

    mp_arr = Array(c.c_double, n*n)
    arr = np.frombuffer(mp_arr.get_obj())

    pairs = ((i, j) for i, j in combinations_with_replacement(range(n), 2))
    pair_chunks = [
        islice(p, i, None, n_processes)
        for i, p in enumerate(tee(pairs, n_processes))
    ]

    def process_pairs(pair_chunk, arr):
        b = arr.reshape((n,n))
        for x in pair_chunk:
            obj1 = pix_chunk[x[0]]
            obj2 = pix_chunk[x[1]]
            difference = compare_objects(obj1,obj2)
            b[x[0],x[1]] = difference
            b[x[1],x[0]] = difference

    processes = [
        Process(target=process_pairs, args=[pair_chunk, arr])
        for pair_chunk in pair_chunks
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    msa = arr.reshape((n,n))

    # comprueba si "obj" pertenece a algun grupo de "groups"
    def belongs_to_group(obj, groups):
        for group in groups:
            for element in group:
                if obj == element:
                    return True
        return False

    for i, row in enumerate(msa):
        group = []
        if belongs_to_group(i+chunk*chunk_size, groups):
            continue
        else:
            group.append(i+chunk*chunk_size)
        for j, element in enumerate(row):
            if i != j and element <= beta:
                group.append(j+chunk*chunk_size)
        groups.append(group)

    chunk+=1

group_colors = []
for group in groups:
    group_length = len(group)
    if group_length <= 1:
        continue
    fm = 0
    am = 0
    bm = 0
    for color in group:
        fm += pix_list_rgb[color][0]
        am += pix_list_rgb[color][1]
        bm += pix_list_rgb[color][2]
    media = (fm//group_length, am//group_length, bm//group_length)
    group_colors.append((media, group))


for color_p in group_colors:
    for pixel_index in color_p[1]:
        pix_list_rgb[pixel_index] = color_p[0]

im2 = Image.new(im.mode, im.size)
im2.putdata(pix_list_rgb)
im2.save('test_{size}px_{beta}.png'.format(size=im.size[0], beta=beta))