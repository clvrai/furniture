from lxml import etree
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=None)
config, unparsed = parser.parse_known_args()

tree = etree.parse(config.path)
root = tree.getroot()
etree.indent(root, space="  ")
et = etree.ElementTree(root)
et.write(config.path, pretty_print=True)
