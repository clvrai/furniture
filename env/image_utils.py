""" Helper functions to convert segmentation map to color image. """

def convert_color(image, from_color, to_color):
    """
    Converts pixels of value @from_color to @to_color in @image.

    Args:
        image: a 3-dim numpy array represents a RGB image
        from_color: a list of length 3 representing a color
        to_color: a list of length 3 representing a color
    """
    image = image.copy()
    red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]

    if from_color is None:
        from_red, from_green, from_blue = \
                image[:,:,0].max(), image[:,:,1].max(), image[:,:,2].max()
        mask = (red == from_red) & (green == from_green) & (blue == from_blue)
    else:
        mask = (red == from_color[0]) & (green == from_color[1]) & (blue == from_color[2])
    image[:,:,:3][mask] = to_color
    return image


def color_segmentation(segmentation):
    """
    Converts a segmentation map to a color image.

    Args:
        segmentation: a 3-dim numpy array represents a RGB image

    Returns:
        colored segmentation map
    """
    color_map = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                 [0, 255, 255], [255, 0, 255], [255, 255, 255], [100, 100, 100],
                 [50, 50, 50], [180, 50, 100], [100, 180, 50], [200, 100, 0],
                 [100, 200, 0], [0, 100, 200], [0, 200, 100], [150, 150, 150],
                 [200, 200, 200]]

    color_seg = segmentation.copy()
    for i in range(len(color_map)):
        color_seg = convert_color(color_seg, [i, i, i], color_map[i])
    return color_seg

