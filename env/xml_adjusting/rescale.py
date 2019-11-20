import xml.etree.ElementTree as ET
import sys
import argparse

def str2bool(v):
    return v.lower() == 'true'
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../models/assets/objects/table_klubbo_0743.xml')
parser.add_argument('--mult', type=float, default=1)
config, unparsed = parser.parse_known_args()

tree = ET.parse(config.path) # Path to input file
root = tree.getroot()
mult = config.mult

for mesh in root.find('asset'):
	if mesh.tag == 'mesh':
		mesh_scale = mesh.attrib['scale'].split(' ')
		mesh_scale = [str(float(i)*mult) for i in mesh_scale]
		upt_mesh_scale = ' '.join(mesh_scale)
		mesh.set('scale', upt_mesh_scale)

for body in root.find('worldbody'):
	body_pos = body.attrib['pos'].split(' ')
	body_pos = [str(float(i)*mult)[:6] for i in body_pos]
	upt_body_pos = ' '.join(body_pos)
	body.set('pos', upt_body_pos)
	for child in body.getiterator():
		if child.tag == 'site':
			site_pos = child.attrib['pos'].split(' ')
			site_pos = [str(float(i)*mult) for i in site_pos]
			upt_site_pos = ' '.join(site_pos)
			child.set('pos', upt_site_pos)
			size = child.attrib['size']
			if ' ' in size: #sometimes size is not a scalar
				size_pos = child.attrib['size'].split(' ')
				size_pos = [str(float(i)*mult) for i in size_pos]
				upt_size_pos = ' '.join(size_pos)
				child.set('size', upt_size_pos)
			else:
				upt_size = str(mult*float(size))
				child.set('size', upt_size)
		elif child.tag == 'geom':
			if 'pos' in child.keys():
				geom_pos = child.attrib['pos'].split(' ')
				geom_pos = [str(float(i)*mult) for i in geom_pos]
				upt_geom_pos = ' '.join(geom_pos)
				child.set('pos', upt_geom_pos)
			if 'size' in child.keys():
				size = child.attrib['size']
				if ' ' in size: #sometimes size is not a scalar
					size_pos = child.attrib['size'].split(' ')
					size_pos = [str(float(i)*mult) for i in size_pos]
					upt_size_pos = ' '.join(size_pos)
					child.set('size', upt_size_pos)
				else:
					upt_size = str(mult*float(size))
					child.set('size', upt_size)

tree.write('out.xml', encoding='UTF-8')
