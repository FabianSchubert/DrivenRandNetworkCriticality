#!/usr/bin/env python3

from PIL import Image
import sys

nx = int(sys.argv[-2])
output = sys.argv[-1]

images = [[]]

for k in range(1,len(sys.argv)-2):
    images[-1].append(Image.open(sys.argv[k]))
    if len(images[-1]) == 2:
        images.append([])

if len(images[-1]) == 0:
    del images[-1]

comp_width = 0
comp_height = 0

rowheights = []

for imgrow in images:
    rowwidth = 0.
    rowheight = 0.
    for img in imgrow:
        rowwidth +=img.size[0]
        if img.size[1] > rowheight:
            rowheight = img.size[1]
    rowheights.append(rowheight)
    comp_height += rowheight
    if rowwidth > comp_width:
        comp_width = rowwidth

comp_width = int(comp_width)
comp_height = int(comp_height)

images_box = []

current_y = 0

for k in range(len(images)):
    imgrow = images[k]
    images_box.append([])
    current_x = 0
    for img in imgrow:
        images_box[-1].append([current_x,current_y,current_x+img.size[0],current_y+img.size[1]])
        current_x += img.size[0]
    current_y += rowheights[k]

#print(images_pos)

im_comp = Image.new('RGB', (comp_width,comp_height))

for k in range(len(images)):
    for l in range(len(images[k])):
        im_comp.paste(images[k][l],images_box[k][l])

im_comp.save(output)
'''
comp_size = (int(textwidth*300),int(6.*300.))

im_comp = Image.new('RGBA', comp_size, color=(255,255,255,255))

images = [Image.open("../plots/act_trans.png"),
        Image.open("../plots/gain.png"),
        Image.open("../plots/var.png"),
        Image.open("../plots/eig.png"),
        Image.open("../plots/act.png")]

images_scaled = []

scaled_img_width = int(comp_size[0]/2)

for img in images:

    img_new_size =  (scaled_img_width,int(img.size[1]*scaled_img_width/img.size[0]))
    images_scaled.append(img.resize(img_new_size,resample=Image.BICUBIC))

for k in range(2):
    for l in range(2):
        box = [0,0,0,0]
        box[0] = int(l*comp_size[0]/2)
        box[1] = int(k*images_scaled[2*k+l].size[1])
        box[2] = box[0] + int(comp_size[0]/2)
        box[3] = box[1] + int(images_scaled[2*k+l].size[1])

        im_comp.paste(images_scaled[2*k+l],box)

box = [0,0,0,0]
box[0] = 0
box[1] = int(2*images_scaled[4].size[1])
box[2] = int(comp_size[0]/2)
box[3] = box[1] + int(images_scaled[4].size[1])

im_comp.paste(images_scaled[4],box)

im_comp.save("../plots/im_comp.png")
'''
