import xml.etree.ElementTree as ET
import sys
import argparse
from util import str2floatlist

def str2bool(v):
    return v.lower() == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='env/models/assets/objects/bench_bjoderna_0208.xml')
parser.add_argument('--translate', type=str2floatlist, default=[0,0,0])
config, unparsed = parser.parse_known_args()

tree = ET.parse(config.path) # Path to input file
root = tree.getroot()
translate = config.translate

for body in root.find('worldbody'):
	body_pos = body.attrib['pos'].split(' ')
	body_pos = [str(float(j) + translate[i]) for i,j in enumerate(body_pos)]
	body.set('pos', ' '.join(body_pos))

tree.write('out.xml', encoding='UTF-8')
