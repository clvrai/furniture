import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import sys
import argparse
import colorsys
import re


# def _get_colors(num_colors):
#     colors=[]
#     for i in np.arange(0., 360., 360. / num_colors):
#         hue = i/360.
#         lightness = (50 + np.random.rand() * 10)/100.
#         saturation = (90 + np.random.rand() * 10)/100.
#         rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
#         rgba = str(tuple([round(x,4) for x in rgb])).strip('()') + ', 0.3'
#         colors.append(rgba)
#     return colors, rgb


def _get_colors(num_colors):
    colors = (
        "0.9019 0.0980 0.2941 0.3",
        "0.2352 0.7058 0.2941 0.3",
        "1.0 0.8823 0.0980 0.3",
        "0.2627 0.3882 0.8470 0.3",
        "0.9607 0.5098 0.1921 0.3",
        "0.5686 0.1176 0.7058 0.3",
        "0.2745 0.9411 0.9411 0.3",
        "0.9411 0.1960 0.9019 0.3",
        "0.7372 0.9647 0.0470 0.3",
        "0.9803 0.7450 0.7450 0.3",
        "0.0 0.5019 0.5019 0.3",
        "0.9019 0.7450 1.0 0.3",
        "0.6039 0.3882 0.1411 0.3",
        "1.0 0.9803 0.7843 0.3",
        "0.5019 0.0 0.0 0.3",
        "0.6666 1.0 0.7647 0.3",
        "0.5019 0.5019 0.0 0.3",
        "1.0 0.8470 0.6941 0.3",
        "0.0 0.0 0.4588 0.3",
        "0.5019 0.5019 0.5019 0.3",
        "1.0 1.0 1.0 0.3",
        "0.0 0.0 0.0 0.3",
    )
    return colors[0:num_colors]


def str2bool(v):
    return v.lower() == "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="../models/assets/objects/in_progress/complete/bench_bjoderna_0208/bench_bjoderna_0208.xml",
    )
    parser.add_argument("--add_welds", type=str2bool, default=True)
    parser.add_argument("--add_colors", type=str2bool, default=True)
    config, unparsed = parser.parse_known_args()
    tree = ET.parse(config.path)  # Path to input file
    root = tree.getroot()
    equality = root.find("equality")

    print(config.path)

    # get count of conn_sites, and get map of groups->bodies
    bodymap = dict()
    connections = set()

    # find group pairs
    num_colors = 0
    for body in root.find("worldbody"):
        for child in body.iter():
            if child.tag == "site" and re.search("conn_site", child.attrib["name"]):
                num_colors += 1
                groupPair = child.attrib["name"].split(",")[0]
                groupNames = groupPair.split("-")
                group1 = groupNames[0]
                group2 = groupNames[1]
                groupPair2 = group2 + "-" + group1
                if group1 not in bodymap.keys():
                    bodies = set()
                    bodies.add(body)
                    bodymap[group1] = bodies
                else:
                    bodymap[group1].add(body)
                if groupPair not in connections and groupPair2 not in connections:
                    connections.add(groupPair)

    if config.add_welds == True:
        for groupPair in connections:
            groupNames = groupPair.split("-")
            group1 = groupNames[0]
            group2 = groupNames[1]
            # n*m welds needed for n bodies in group1 and m bodies in group2
            for body1 in bodymap[group1]:
                for body2 in bodymap[group2]:
                    weld = ET.SubElement(equality, "weld")
                    weld.set("active", "false")
                    weld.set("body1", body1.attrib["name"])
                    weld.set("body2", body2.attrib["name"])
                    weld.set("solimp", "1 1 0.5")
                    weld.set("solref", "0.01 0.3")

    if config.add_colors == True:
        num_colors = int(num_colors / 2)
        colors = _get_colors(num_colors)
        # for color in colors:
        # 	print(color)
        i = 0
        colormap = dict()
        for body in root.find("worldbody"):
            for child in body.iter():
                if child.tag == "site" and re.search("conn_site", child.attrib["name"]):
                    groupPair = child.attrib["name"].split(",")[0]
                    if groupPair not in colormap:
                        groupNames = groupPair.split("-")
                        group1 = groupNames[0]
                        group2 = groupNames[1]
                        colormap[groupPair] = colors[i]
                        groupPair2 = group2 + "-" + group1
                        colormap[groupPair2] = colors[i]
                        i += 1
                    # change color of conn_site
                    child.set("rgba", colormap[groupPair])

    tree.write(config.path, encoding="UTF-8")


if __name__ == "__main__":
    main()
