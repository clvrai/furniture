import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import sys
import argparse
import colorsys

def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgba = str(tuple([round(x,4) for x in rgb])).strip('()') + ', 0.3'
        colors.append(rgba)
    return colors, rgb


def str2bool(v):
    return v.lower() == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../models/assets/objects/in_progress/complete/bench_bjoderna_0208/bench_bjoderna_0208.xml')
config, unparsed = parser.parse_known_args()
tree = ET.parse(config.path) # Path to input file
root = tree.getroot()

num_colors = 0
for body in root.find('worldbody'):
	for child in body.getiterator():
		if child.tag == 'site' and child.attrib['name'].endswith('conn_site'):
			num_colors +=1

num_colors = int(num_colors/2)
colors, palette = _get_colors(num_colors)

# current_palette = sns.color_palette(None, n_colors=20)
# colors = []
# for color in current_palette:
# 	print(color)
# 	colors.append(str(tuple([round(x,4) for x in color])).strip('()') + ', 0.3')

for color in colors:
	print(color)

equality = root.find("equality")

i = 0
colormap = dict()
for body in root.find('worldbody'):
	for child in body.getiterator():
		if child.tag == 'site' and child.attrib['name'].endswith('conn_site'):
			sitename = child.attrib['name'].split(',')[0]
			if sitename not in colormap:
				bodynames = sitename.split('-')
				body1 = bodynames[0]
				body2 = bodynames[1]
				# add weld to XML
				weld = ET.SubElement(equality, 'weld')
				weld.set('active', 'false')
				weld.set('body1', body1)
				weld.set('body2', body2)
				weld.set('solimp', '1 1 0.5')
				weld.set('solref', '0.01 0.3')
				colormap[sitename] = colors[i]
				sitename2 = body2+'-'+body1
				colormap[sitename2] = colors[i] 
				i+=1
			# change color of conn_site
			child.set('rgba', colormap[sitename]) 

tree.write(config.path, encoding='UTF-8')