import argparse
import sys
import xml.etree.ElementTree as ET

from pyquaternion import Quaternion


def str2bool(v):
    return v.lower() == "true"


def str2floatlist(v):
    return [float(x) for x in v.split(",")]


def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="../models/assets/objects/bench_bjoderna_0208.xml"
    )
    parser.add_argument("--mult", type=float, default=1)
    parser.add_argument("--translate", type=str2floatlist, default=[])
    parser.add_argument("--rotate", type=str2floatlist, default=[])
    parser.add_argument("--outpath", type=str, default="out.xml")
    parser.add_argument("--write", type=str2bool, default=False)
    args, unparsed = parser.parse_known_args()
    return args


def rescale(tree, root, mult, outpath="out.xml", translate=[], rotate=[], write=False):
    for mesh in root.find("asset"):
        if (
            mesh.tag == "mesh"
            and "name" in mesh.attrib
            and "part" in mesh.attrib["name"]
        ):
            mesh_scale = mesh.attrib["scale"].split(" ")
            mesh_scale = [str(float(i) * mult) for i in mesh_scale]
            upt_mesh_scale = " ".join(mesh_scale)
            mesh.set("scale", upt_mesh_scale)

    for body in root.find("worldbody"):
        if "name" in body.attrib and "_part" in body.attrib["name"]:
            body_pos = body.attrib["pos"].split(" ")
            body_quat = [1, 0, 0, 0]
            if "quat" in body.attrib:
                body_quat = body.attrib["quat"].split(" ")
            body_pos = [str(float(i) * mult) for i in body_pos]
            if len(translate) > 0:
                body_pos = [str(float(i) + j) for i, j in zip(body_pos, translate)]
            if len(rotate) > 0:
                w, x, y, z = [float(i) for i in body_quat]
                rotate_quat = Quaternion(rotate) * Quaternion(w, x, y, z)
                body_quat = [str(i) for i in rotate_quat.elements]
                body_quat_s = " ".join(body_quat)
                body.set("quat", body_quat_s)

            upt_body_pos = " ".join(body_pos)
            body.set("pos", upt_body_pos)

            for child in body.getiterator():
                if child.tag == "site":
                    site_pos = child.attrib["pos"].split(" ")
                    site_pos = [str(float(i) * mult) for i in site_pos]
                    upt_site_pos = " ".join(site_pos)
                    child.set("pos", upt_site_pos)
                    size = child.attrib["size"]
                    if " " in size:  # sometimes size is not a scalar
                        size_pos = child.attrib["size"].split(" ")
                        size_pos = [str(float(i) * mult) for i in size_pos]
                        upt_size_pos = " ".join(size_pos)
                        child.set("size", upt_size_pos)
                    else:
                        upt_size = str(mult * float(size))
                        child.set("size", upt_size)
                elif child.tag == "geom":
                    if "pos" in child.keys():
                        geom_pos = child.attrib["pos"].split(" ")
                        geom_pos = [str(float(i) * mult) for i in geom_pos]
                        upt_geom_pos = " ".join(geom_pos)
                        child.set("pos", upt_geom_pos)
                    if "size" in child.keys():
                        size = child.attrib["size"]
                        if " " in size:  # sometimes size is not a scalar
                            size_pos = child.attrib["size"].split(" ")
                            size_pos = [str(float(i) * mult) for i in size_pos]
                            upt_size_pos = " ".join(size_pos)
                            child.set("size", upt_size_pos)
                        else:
                            upt_size = str(mult * float(size))
                            child.set("size", upt_size)
    if write:
        tree.write(outpath, encoding="UTF-8")
    return tree


def rescale_connsite(tree, root, mult):
    for body in root.find("worldbody"):
        if "name" in body.attrib and "_part" in body.attrib["name"]:
            for child in body.getiterator():
                if child.tag == "site" and "name" in child.attrib and "conn_site" in child.attrib["name"]:
                    size = child.attrib["size"]
                    if " " in size:  # sometimes size is not a scalar
                        size_pos = child.attrib["size"].split(" ")
                        size_pos = [str(float(i) * mult) for i in size_pos]
                        upt_size_pos = " ".join(size_pos)
                        child.set("size", upt_size_pos)
                    else:
                        upt_size = str(mult * float(size))
                        child.set("size", upt_size)
    return tree


def rescale_numeric(tree, root, mult, outpath="out.xml", translate=[], rotate=[], write=False):
    for body in root.find("custom"):
        if "name" in body.attrib and "_part" in body.attrib["name"]:
            data = body.attrib["data"].split(" ")
            body_pos = data[:3]
            body_quat = data[3:]
            body_pos = [str(float(i) * mult) for i in body_pos]
            if len(translate) > 0:
                body_pos = [str(float(i) + j) for i, j in zip(body_pos, translate)]
            if len(rotate) > 0:
                w, x, y, z = [float(i) for i in body_quat]
                rotate_quat = Quaternion(rotate) * Quaternion(w, x, y, z)
                body_quat = [str(i) for i in rotate_quat.elements]
                body_quat_s = " ".join(body_quat)
                body.set("quat", body_quat_s)

            upt_body_pos = " ".join(body_pos + body_quat)
            body.set("data", upt_body_pos)
    if write:
        tree.write(outpath, encoding="UTF-8")
    return tree


def main(config):
    tree = ET.parse(config.path)  # Path to input file
    root = tree.getroot()
    rescale_numeric(
        tree,
        root,
        config.mult,
        outpath=config.outpath,
        translate=config.translate,
        rotate=config.rotate,
        write=config.write,
    )


if __name__ == "__main__":
    config = argsparser()
    main(config)
