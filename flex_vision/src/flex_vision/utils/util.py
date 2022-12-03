# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 21:21:28 2020

@author: taeke
"""
import os
import math
from typing import TYPE_CHECKING

# External imports
import cv2
import matplotlib as mpl
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Flex vision imports
from flex_vision import constants
from flex_vision.utils import color_maps

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import typing

ee_color = (255, 150, 0)
grasp_color = (200, 0, 150)
tomato_color = (255, 82, 82)
peduncle_color = (82, 255, 82)  # (0, 150, 30)
background_color = (0, 0, 255)
junction_color = (100, 0, 200)
end_color = (200, 0, 0)
gray_color = (150, 150, 150)

background_layer = 0
bottom_layer = 1  # contours
middle_layer = 5  # tomatoes
peduncle_layer = 6
vertex_layer = 7
high_layer = 8  # arrows, com
top_layer = 10  # junctions, com, text


LINEWIDTH = 3.4  # inch

# Options
params = {'text.usetex': True,
          'font.size': 10,        # controls default text sizes
          # 'legend.fontsize': 10,    # fontsize of the legend
          # 'axes.labelsize': 10,     # fontsize of axis labels
          # 'xtick.labelsize': 10,    # fontsize of x-axis ticks
          # 'ytick.labelsize': 10,    # fontsize of y-axis ticks
          'font.family': 'serif',
          'font.serif': 'Times',
          # 'text.latex.unicode': True,
          # 'pdf.fonttype': 42  # https://jdhao.github.io/2018/01/18/mpl-plotting-notes-201801/
          }
plt.rcParams.update(params)


def create_directory(directory):
    # type: (str) -> None
    """ Make a (nested) directory, if it does not yet exist. 

    Args:
        directory: The directory to create
    """
    if not os.path.isdir(directory):
        print("Creating directory " + directory)
        os.makedirs(directory)


def load_image(image_file, horizontal=True, cv2_flag=cv2.COLOR_BGR2RGB):
    # type: (str, bool, int) -> np.typing.ArrayLike
    """ Load an image from a file, and transpose and/or convert colors if requested.

    Args:
        image_file: The file to load the image from
        horizontal: If set to true, the image will be transposed such that it is oriented horizontally. Defaults to True.
        cv2_flag: CV2 color conversion flag. Defaults to cv2.COLOR_BGR2RGB.

    Returns:
        Image if loaded successfully. Otherwise None.
    """
    if not os.path.isfile(image_file):
        print('Cannot load RGB data from file: ' + image_file + ', as it does not exist!')
        return None

    image = cv2.imread(image_file)
    if image is None:
        print("Failed to load image from path: %s" % image_file)
        return None

    # apply cv2 color conversion if requested
    if cv2_flag is not None:
        image = cv2.cvtColor(image, cv2_flag)

    # transpose image if required
    if horizontal and (image.shape[0] > image.shape[1]):
        image = np.transpose(image, [1, 0, 2])

    return image


def bin2img(binary, dtype=np.uint8, copy=False):
    # type (np.typing.ArrayLike, np.typing.DTypeLike, bool) -> np.typing.ArrayLike
    """ Convert a binary array to the desired data type.

    Args:
        binary: The binary array
        dtype: Numpy data type. Defaults to np.uint8.
        copy: Copy the array if True. Defaults to False.

    Returns:
        The image in the desired data type.
    """
    return binary.astype(dtype, copy=copy) * np.iinfo(dtype).max


def img2bin(image, copy=False):
    # type (np.typing.ArrayLike, bool) -> np.typing.ArrayLike
    """ Convert an image to a binary array.

    Args:
        image: The image.
        copy: Copy the array if True. Defaults to False.

    Returns:
        The binary array.
    """
    return image.astype(bool, copy=copy)


def change_brightness(image, brightness, copy=True):
    # type: (np.typing.ArrayLike, float, bool) -> np.typing.ArrayLike
    """ Change image brightness.

    Args:
        image: The image.
        brightness: The brightness values [-1.0, 1.0], A positive number increases the brightness, and negative value 
          reduces the brightness.
        copy: Copy the array if True. Defaults to True.

    Raises:
        ValueError: if the brightness is not within the specified range

    Returns:
        The image with changed brightness.
    """
    if copy:
        image = image.copy()

    if 0 <= brightness < 1:
        return image + ((255 - image) ** brightness).astype(np.uint8)

    if -1 < brightness < 0:
        return image - (image ** -brightness).astype(np.uint8)

    raise ValueError("Brightness of %f is invalid, please provide a value within range [-1.0, 1.0]" % brightness)


def angular_difference(alpha, beta):
    # type: (float, float) -> float
    """ Compute the difference between two angles.

    Args:
        alpha: An angle in radians.
        beta: An angle in radians.

    Returns:
        The angular difference in radians.
    """
    return abs(math.atan2(math.sin(alpha - beta), math.cos(alpha - beta)))


def remove_blobs(image):
    # type: (np.typing.ArrayLike) -> np.typing.ArrayLike
    """ Remove blobs from an image, such that only the largest blob remains.

    Args:
        image: The input image.

    Returns:
        The filtered image.
    """
    filtered_image = np.zeros(image.shape[:2], image.dtype)     # initialize outgoing image

    # The indexing is essential to make it work with various cv2 versions (see: https://stackoverflow.com/a/56142875)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(filtered_image, [cnt], -1, np.iinfo(image.dtype).max, cv2.FILLED)

    return cv2.bitwise_and(image, filtered_image)


def stack_segments(image, background, tomato, peduncle, use_image_colours=True):
    # type: (np.typing.ArrayLike, np.typing.ArrayLike, np.typing.ArrayLike, np.typing.ArrayLike, bool) -> np.typing.ArrayLike
    """ Stack segments.

    Args:
        image: The image.
        background: The background segment.
        tomato: The tomato segment.
        peduncle: The peduncle segment.
        use_image_colours: If true use the average color of the segment in the original image, otherwise use the default 
          colors. Defaults to True.

    Returns:
        The stacked image segments
    """
    # segment labels
    backgroundLabel = 0
    tomatoLabel = 1
    peduncleLabel = 2

    # label pixels
    pixelLabel = np.zeros(image.shape[:2], dtype=np.int8)
    pixelLabel[background > 0] = backgroundLabel
    pixelLabel[cv2.bitwise_and(tomato, cv2.bitwise_not(peduncle)) > 0] = tomatoLabel
    pixelLabel[peduncle > 0] = peduncleLabel

    # get class colors
    if use_image_colours:
        color_background = np.uint8(np.mean(image[pixelLabel == backgroundLabel], 0))
        color_tomato = np.uint8(np.mean(image[pixelLabel == tomatoLabel], 0))
        color_peduncle = np.uint8(np.mean(image[pixelLabel == peduncleLabel], 0))

    else:
        color_background = background_color
        color_tomato = tomato_color
        color_peduncle = peduncle_color

    color = np.vstack([color_background, color_tomato, color_peduncle]).astype(np.uint8)
    res = color[pixelLabel.flatten()]
    return res.reshape(image.shape)


def grey_2_rgb(image, vmin=0, vmax=255, cmap=mpl.cm.hot):  # pylint: disable=no-member
    # type: (np.typing.ArrayLike, int, int, mpl.colors.LinearSegmentedColormap) -> np.typing.ArrayLike
    """ Convert a grey scale image to an RGB image.

    Args:
        image: The grey image
        vmin: The minimum value of yhe output image. Defaults to 0.
        vmax: The maximum value of yhe output image. Defaults to 255.
        cmap: The color map to use. Default to mpl.cm.hot.

    Returns:
        The RGB image.
    """
    mapping = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    return (vmax * mapping.to_rgba(image)[:, :, 0:3]).astype(np.uint8)


def save_img(image,                           # type: np.typing.ArrayLike
             pwd,                           # type: str
             name,                          # type: str
             dpi=constants.DPI,             # type: int
             title="",                      # type: str
             title_size=20,                 # type: int
             color_map='plasma',            # type: str
             vmin=None,                     # type: typing.Optional[int]
             vmax=None                      # type: typing.Optional[int]
             ):
    # type (...) -> None
    """ Save image

    Args:
        image: The image
        pwd: The path to save the image to
        name: The file name.
        dpi: The DPI. Defaults to constants.DPI
        title: The title. Defaults to ""
        title_size: The title size. Defaults to 20.
        color_map: The color map. Defaults to plasma
        vmin: The minimum value of yhe output image. Defaults to None.
        vmax: The maximum value of yhe output image. Defaults to None.
    """
    plt.rcParams["savefig.format"] = constants.SAVE_EXTENSION[1:]  # remove leading "."
    plt.rcParams["savefig.bbox"] = 'tight'
    plt.rcParams['axes.titlesize'] = title_size

    if color_map == 'HSV':
        color_map = color_maps.hsv_color_scale()
    elif color_map == 'Lab':
        color_map = color_maps.lab_color_scale()

    sizes = np.shape(image)
    fig = plt.figure()
    fig.set_size_inches(float(sizes[1])/float(sizes[0]), 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, cmap=color_map, vmin=vmin, vmax=vmax)
    # plt.axis('off')
    if title:
        plt.title(title)

    # https://stackoverflow.com/a/27227718
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # make dir if it does not yet exist
    create_directory(pwd)
    fig.savefig(os.path.join(pwd, name), dpi=dpi)
    plt.close(fig)


def save_fig(fig, pwd, name, dpi=constants.DPI, no_ticks=True, ext=constants.SAVE_EXTENSION):
    """ Save figure.

    Args:
        fig: The figure
        pwd: The path
        name: The file name
        dpi: The dots per image. Defaults to constants.DPI.
        no_ticks: If set to True, no ticks will be shown. Defaults to True.
        ext: The file extension. Defaults to constants.SAVE_EXTENSION.
    """
    # eps does not support transparancy
    # plt.rcParams["savefig.format"] = ext

    # for ax in fig.get_axes():
    #     pass
    #     # ax.label_outer()

    if no_ticks:
        for ax in fig.get_axes():
            ax.set_yticklabels([])
            ax.set_xticklabels([])

    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # make dir if it does not yet exist
    create_directory(pwd)
    fig.savefig(os.path.join(pwd, name + ext), dpi=dpi)  # , bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_segments(img_rgb, background, tomato, peduncle, fig=None, show_background=False, pwd=None,
                  use_image_colours=True, show_axis=False, name=None, title=None, alpha=0.4, linewidth=0.5, ncols=1):
    """
        alpha: trasparancy of segments, low value is more transparant!
    """

    if fig is None:
        fig = plt.figure()

    img_segments = stack_segments(img_rgb, background, tomato, peduncle, use_image_colours=use_image_colours)
    added_image = cv2.addWeighted(img_rgb, 1 - alpha, img_segments, alpha, 0)
    plot_image(added_image, show_axis=show_axis, ncols=ncols)

    # plot all contours
    if show_background:
        add_contour(background, color=background_color, linewidth=linewidth)
    add_contour(tomato, color=tomato_color, linewidth=linewidth)
    add_contour(peduncle, color=peduncle_color, linewidth=linewidth)

    if title:
        plt.title(title)

    if pwd is not None:
        save_fig(fig, pwd, name)

    return fig


def plot_image(img, show_axis=False, animated=False, nrows=1, ncols=1):
    """
        plot image
    """
    sizes = np.shape(img)
    fig = plt.figure(dpi=constants.DPI)
    fig.set_size_inches(LINEWIDTH, LINEWIDTH * float(sizes[0]) / float(sizes[1]), forward=False)

    # if multiple axes are desired we add them using gridspec
    if (nrows > 1) or (ncols > 1):
        # add axs   left  bott  width height
        rect_bot = [0.06, 0.24, 0.82, 0.63]
        rect_top = [0.85, 0.24, 0.02, 0.63]
        plt.axes(rect_bot)
        plt.axes(rect_top)

        # set first axs active
        plt.sca(plt.gcf().get_axes()[0])

    if not show_axis:
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

    plt.imshow(img, animated=animated)

    if not show_axis:
        clear_axis()


def clear_axis():
    plt.tight_layout()
    plt.axis('off')

    # https://stackoverflow.com/a/27227718
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())


def plot_truss(img_rgb=None, tomato=None, peduncle=None):
    if img_rgb is not None:
        plt.figure()
        plot_image(img_rgb)

    if tomato:
        add_circles(tomato['centers'], radii=tomato['radii'], fc=tomato_color, ec=tomato_color)

    if peduncle:
        add_lines(peduncle['centers'], peduncle['angles'],
                  lengths=peduncle['length'], color=peduncle_color, linewidth=5)


def plot_features(img_rgb=None, tomato=None, peduncle=None, grasp=None,
                  alpha=0.4, zoom=False, pwd=None, file_name=None):

    if img_rgb is not None:
        plot_image(img_rgb)

    fig = plt.gcf()

    if zoom:
        tom_linestyle = (0, (10, 10))
        tom_width = 1.5
        com_radius = 10
        junc_radius = 8
    else:
        tom_linestyle = (0, (5, 5))
        tom_width = 1.5
        com_radius = 6
        junc_radius = 8

    if tomato:
        add_circles(tomato['centers'], radii=tomato['radii'], fc=tomato_color, linewidth=tom_width, alpha=alpha,
                    linestyle=tom_linestyle)

        if 'com' in tomato.keys():
            add_com(tomato['com'], radius=com_radius)

    if peduncle:
        add_circles(peduncle['junctions'], radii=junc_radius, fc=junction_color, linewidth=0.5, zorder=top_layer)

    if grasp:
        col = grasp['col']
        row = grasp['row']
        angle = grasp['angle']
        plot_grasp_location([[col, row]], angle, finger_width=20, finger_thickness=15, linewidth=2)

    if pwd is not None:
        save_fig(fig, pwd, file_name)


def plot_features_result(img_rgb, tomato_pred=None, peduncle=None, grasp=None, alpha=0.5, linewidth=1.5, zoom=False,
                         pwd=None, name=None, title="", fig=None):

    plot_image(img_rgb)
    fig = plt.gcf()
    if zoom:
        tom_linestyle = (0, (10, 10))
        com_radius = 10
        junc_radius = 8
    else:
        tom_linestyle = (0, (5, 5))
        com_radius = 6
        junc_radius = 8

    if tomato_pred:
        add_circles(tomato_pred['true_pos']['centers'], radii=tomato_pred['true_pos']['radii'], fc=tomato_color,
                    linewidth=linewidth, alpha=alpha, linestyle=tom_linestyle)
        add_circles(tomato_pred['false_pos']['centers'], radii=tomato_pred['false_pos']['radii'], fc=tomato_color,
                    linewidth=linewidth, alpha=alpha, linestyle=tom_linestyle)
        add_com(tomato_pred['com'], radius=com_radius)

    if peduncle:
        add_circles(peduncle['false_pos']['centers'], radii=junc_radius, ec=(255, 0, 0), linewidth=0.5, alpha=0,
                    zorder=top_layer)
        add_circles(peduncle['true_pos']['centers'], radii=junc_radius, fc=junction_color, linewidth=0.5,
                    zorder=top_layer)

    if grasp:
        col = grasp['col']
        row = grasp['row']
        angle = grasp['angle']
        plot_grasp_location([[col, row]], angle, finger_width=20, finger_thickness=15, linewidth=2)

    if title:
        plt.title(title)

    if pwd is not None:
        save_fig(fig, pwd, name)


def plot_timer(timer_dict, N=1, threshold=0, ignore_key=None, pwd=None, name='time', title='time', startangle=-45):
    for key in timer_dict.keys():

        # remove ignored keys
        if key == ignore_key:
            del timer_dict[key]
            continue

        # remove empty keys
        if timer_dict[key] == []:
            del timer_dict[key]

    values = np.array(timer_dict.values())

    if len(values.shape) == 1:
        values = values / N
    else:
        values = np.mean(values, axis=1)

    labels = np.array(timer_dict.keys())

    values_rel = values / np.sum(values)
    i_keep = (values_rel > threshold)
    i_remove = np.bitwise_not(i_keep)

    # if everything is put under others
    if np.all(i_remove is True):
        print("No time to plot!")
        return

    labels_keep = labels[i_keep].tolist()
    values_keep = values[i_keep].tolist()

    if np.any(i_remove is True):
        remains = np.mean(values[i_remove])
        values_keep.append(remains)
        labels_keep.append('others')

    l = zip(values_keep, labels_keep)
    l.sort()
    values_keep, labels_keep = zip(*l)

    donut(values_keep, labels_keep, pwd=pwd, name=name, title=title, startangle=startangle)


def donut(data, labels, pwd=None, name=None, title=None, startangle=-45):
    data = np.array(data)
    data_rel = data / sum(data) * 100

    text = []
    separator = ': '
    for label, value, value_rel in zip(labels, data, data_rel):
        text.append(label + separator + str(int(round(value_rel))) + '% (' + str(int(round(value))) + ' ms)')

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    wedges, _ = ax.pie(data, wedgeprops=dict(width=0.5), startangle=startangle)

    bbox_props = dict(boxstyle="round,pad=0.3", fc=[0.92, 0.92, 0.92], lw=0)  # square, round
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center", fontsize=15)

    y_scale = 1.8

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(text[i], xy=(x, y), xytext=(1.35 * np.sign(x), y_scale * y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title(title, fontsize=20)
    plt.tight_layout()

    if pwd is not None:
        save_fig(fig, pwd, name, no_ticks=False)


def plot_grasp_location(loc, angle, finger_width=20, finger_thickness=10, finger_dist=None, linewidth=1, pwd=None,
                        name=None, title=''):
    """
        angle in rad
    """

    if isinstance(loc, (list, tuple, np.matrix)):
        loc = np.array(loc, ndmin=2)

    if angle is None:
        return

    if len(loc.shape) > 1:
        if loc.shape[0] > loc.shape[1]:
            loc = loc.T

        loc = loc[0]

    if (loc[0] is None) or (loc[1] is None):
        return

    rot_angle = angle + np.pi / 2
    if finger_dist is not None:
        right_origin, left_origin = compute_line_points(loc, rot_angle, finger_dist)

    else:
        left_origin = loc
        right_origin = loc

    R = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])

    xy = np.array([[-finger_thickness], [-finger_width/2]])
    xy_rot = np.matmul(R, xy) + np.expand_dims(left_origin, axis=1)

    add_rectangle(xy_rot, finger_thickness, finger_width, angle=np.rad2deg(rot_angle), ec=ee_color,
                  fc=ee_color, alpha=0.4, linewidth=linewidth, zorder=middle_layer)

    if finger_dist is not None:
        add_rectangle(xy_rot, 2*finger_thickness + finger_dist, finger_width, angle=np.rad2deg(rot_angle), ec=grasp_color,
                      fc=grasp_color, alpha=0.4, linewidth=linewidth, linestyle='-', zorder=bottom_layer)

    xy = np.array([[0], [-finger_width/2]])
    xy_rot = np.matmul(R, xy) + np.expand_dims(right_origin, axis=1)

    add_rectangle(xy_rot, finger_thickness, finger_width, angle=np.rad2deg(rot_angle), ec=ee_color,
                  fc=ee_color, alpha=0.4, linewidth=linewidth, zorder=middle_layer)

    if title:
        plt.title(title)

    if pwd is not None:
        save_fig(plt.gcf(), pwd, name)


def plot_error(tomato_pred, tomato_act, error,
               pwd=None,
               name=None,
               use_mm=False,
               title=None,
               dpi=constants.DPI,
               ext=constants.SAVE_EXTENSION):

    fig = plt.gcf()
    ax = plt.gca()

    if title:
        plt.title(title)

    if use_mm:
        unit = 'mm'
    else:
        unit = 'px'

    n_true_pos = len(tomato_pred['true_pos']['centers'])
    n_false_pos = len(tomato_pred['false_pos']['centers'])
    n_false_neg = len(tomato_act['false_neg']['centers'])
    if 'com' in tomato_pred.keys():
        n_com = len(tomato_pred['com'])
    else:
        n_com = 0

    centers = []
    centers.extend(tomato_pred['true_pos']['centers'])
    centers.extend(tomato_pred['false_pos']['centers'])
    centers.extend(tomato_act['false_neg']['centers'])
    if 'com' in tomato_pred.keys():
        centers.extend([tomato_pred['com']])

    labels = []
    labels.extend(['true_pos'] * n_true_pos)
    labels.extend(['false_pos'] * n_false_pos)
    labels.extend(['false_neg'] * n_false_neg)
    labels.extend(['com'] * n_com)

    error_centers = []
    error_centers.extend(error['centers'])
    error_centers.extend(n_false_pos * [None])
    error_centers.extend(n_false_neg * [None])
    if 'com' in tomato_pred.keys():
        error_centers.append(error['com'])

    if 'radii' in error.keys():
        error_radii_val = error['radii']
    else:
        error_radii_val = [None] * n_true_pos
    error_radii = []
    error_radii.extend(error_radii_val)
    error_radii.extend(n_false_pos * [None])
    error_radii.extend(n_false_neg * [None])
    error_radii.extend(n_com * [None])

    # sort based on the y location of the centers
    zipped = zip(centers, error_centers, error_radii, labels)
    zipped.sort(key=lambda x: x[0][1])
    centers, error_centers, error_radii, labels = zip(*zipped)

    # default bbox style
    bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="w", lw=0)
    kw_default = dict(arrowprops=dict(arrowstyle="-", linewidth=0.5),
                      bbox=bbox_props, va="center", size=8, color='k')

    y_lim = ax.get_ylim()
    h = y_lim[0] - y_lim[1]
    x_lim = ax.get_xlim()
    w = x_lim[1] - x_lim[0]

    # n determines the spacing of the baxes from the top of the image
    if 'radii' in error.keys():
        n = 0.5

    else:
        n = 3

    y_increment = h / (len(centers) + n)
    y_text = (n/2.0 + 0.5) * y_increment  # 1.0/n* h

    # kw_list = []
    for center, error_center, error_radius, label in zip(centers, error_centers, error_radii, labels):

        # copy default style
        kw = deepcopy(kw_default)

        if label == 'true_pos':

            center_error = int(round(error_center))
            if error_radius is not None:
                radius_error = int(round(error_radius))
                text = 'loc: {c:d}{u:s} \nr: {r:d}{u:s}'.format(c=center_error, r=radius_error, u=unit)
            else:
                text = 'loc: {c:d}{u:s}'.format(c=center_error, u=unit)
            arrow_color = mpl.colors.colorConverter.to_rgba('w', alpha=1)

        elif label == 'com':
            center_error = int(round(error_center))
            text = 'com: {c:d}{u:s}'.format(c=center_error, u=unit)
            kw['bbox']['fc'] = 'k'
            kw['bbox']['ec'] = 'k'
            kw['color'] = 'w'
            arrow_color = mpl.colors.colorConverter.to_rgba('w', alpha=1)

        elif label == 'false_pos':
            text = 'false positive'
            kw['bbox']['fc'] = 'r'
            kw['bbox']['ec'] = 'r'
            arrow_color = 'r'

        elif label == 'false_neg':
            text = 'false negative'
            kw['bbox']['ec'] = 'lightgrey'
            kw['bbox']['fc'] = 'lightgrey'
            arrow_color = 'lightgrey'

        y = center[1]
        x = center[0]

        if x <= 0.35 * w:
            x_text = 0.6 * w
        elif x <= 0.5 * w:
            x_text = 0.05 * w
        elif x <= 0.65 * w:
            x_text = 0.8 * w
        else:
            x_text = 0.2 * w  # w

        x_diff = x_text - x
        y_diff = y_text - y
        if (x_diff > 0 and y_diff > 0) or (x_diff < 0 and y_diff < 0):
            ang = -45
        else:
            ang = 45

        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle, 'color': arrow_color})
        # kw_list.append(kw)
        plt.annotate(text, xy=(x, y), xytext=(x_text, y_text), zorder=high_layer, **kw)  #
        y_text = y_text + y_increment

    if pwd:
        save_fig(fig, pwd=pwd, name=name, dpi=dpi, ext=ext)


def compute_line_points(center, angle, l):
    """
        angle in rad
    """
    col = center[0]
    row = center[1]

    col_start = col + 0.5 * l * np.cos(angle)
    row_start = row + 0.5 * l * np.sin(angle)

    col_end = col - 0.5 * l * np.cos(angle)
    row_end = row - 0.5 * l * np.sin(angle)

    start_point = np.array([col_start, row_start])
    end_point = np.array([col_end, row_end])
    return start_point, end_point


def add_com(center, radius=5):
    """
        center: circle centers expressed in [col, row]
    """
    ax = plt.gca()
    center = np.array(center, ndmin=2)

    pwd = os.path.dirname(__file__)
    pwd_img = os.path.join(pwd, '..', '..', '..', 'images')
    img = mpl.image.imread(os.path.join(pwd_img, 'com.png'))

    com_radius, _, _ = img.shape
    com_zoom = radius / float(com_radius)
    imagebox = mpl.offsetbox.OffsetImage(img, zoom=com_zoom)

    props = dict(alpha=0, zorder=top_layer)
    ab = mpl.offsetbox.AnnotationBbox(imagebox, (center[0, 0], center[0, 1]), pad=0, bboxprops=props)
    ab.set_zorder(high_layer)
    ax.add_artist(ab)


def add_circles(centers, radii=5, fc=(255, 255, 255), ec=(0, 0, 0), linewidth=1, alpha=1.0, linestyle='-', zorder=None,
                pwd=None, name=None):
    """
        centers: circle centers expressed in [col, row]
    """

    if zorder is None:
        zorder = middle_layer

    if isinstance(centers, (list, tuple, np.matrix)):
        centers = np.array(centers, ndmin=2)

    # if a single radius is give, we repeat the value
    if not isinstance(radii, (list, np.ndarray)):
        radii = [radii] * centers.shape[0]

    if len(centers.shape) == 1:
        centers = np.array(centers, ndmin=2)

    # if empty we can not add any circles
    if centers.shape[0] == 0:
        return

    if centers.shape[1] == 0:
        return

    fc = np.array(fc).astype(float) / 255
    ec = np.array(ec).astype(float) / 255

    fc = np.append(fc, alpha)
    ec = np.append(ec, 1)

    # centers should be integers
    centers = np.round(centers).astype(dtype=int)  # (col, row)
    radii = np.round(radii).astype(dtype=int)  # (col, row)

    for center, radius in zip(centers, radii):
        circle = mpl.patches.Circle(center, radius, ec=ec, fc=fc, fill=True, linewidth=linewidth,
                                    linestyle=linestyle, zorder=zorder)

        ax = plt.gca()
        artist = ax.add_artist(circle)

    if pwd is not None:
        save_fig(plt.gcf(), pwd, name)

    return artist


def add_contour(mask, color=(255, 255, 255), linewidth=1, zorder=None):
    if zorder is None:
        zorder = bottom_layer

    color = np.array(color).astype(float) / 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for contour in contours:
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], linestyle='-', linewidth=linewidth, color=color,
                 zorder=zorder)


def add_lines(centers, angles, lengths=20, color=(255, 255, 255), linewidth=1, is_rad=True):
    """
    angle in rad
    """
    if isinstance(centers, (list, tuple)):
        centers = np.array(centers, ndmin=2)

    if len(centers.shape) == 1:
        centers = np.array(centers, ndmin=2)

    if not isinstance(angles, (list, tuple)):
        angles = [angles]

    if not isinstance(color, (str)):
        color = np.array(color).astype(float) / 255

    if not isinstance(lengths, (list, tuple)):
        lengths = [lengths] * len(angles)

    for center, angle, length in zip(centers, angles, lengths):
        if not is_rad:
            angle = angle / 180 * np.pi

        start_point, end_point = compute_line_points(center, angle, length)
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color, linewidth=linewidth,
                 zorder=top_layer)


def add_lines_from_points(start_point, end_point, color=(255, 255, 255), linewidth=1, linestyle='-'):
    if not isinstance(color, (str)):
        color = np.array(color).astype(float) / 255

    plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color, linewidth=linewidth,
             linestyle=linestyle, zorder=top_layer)


def add_arrows(centers, angles, lengths=20, color=(255, 255, 255), linewidth=1, head_width=5, head_length=7, is_rad=True):
    """
    angle in rad
    """
    if isinstance(centers, (list, tuple)):
        centers = np.array(centers, ndmin=2)

    if not isinstance(angles, (list, tuple, np.ndarray)):
        angles = [angles]

    if not isinstance(lengths, (list, tuple, np.ndarray)):
        lengths = [lengths]

    color = np.array(color).astype(float) / 255

    for center, angle, length in zip(centers, angles, lengths):
        if not is_rad:
            angle = angle / 180 * np.pi
        start_point, end_point = compute_line_points(center, angle, length)

        plt.arrow(start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1],
                  color=color, lw=linewidth, head_width=head_width, head_length=head_length, zorder=top_layer)


def add_rectangle(xy, width, height, angle=0, fc=(255, 255, 255), ec=(0, 0, 0), linewidth=1, alpha=1.0, linestyle='-', zorder=None):

    if zorder is None:
        zorder = middle_layer

    fc = np.array(fc).astype(float) / 255
    ec = np.array(ec).astype(float) / 255

    fc = np.append(fc, alpha)
    ec = np.append(ec, 1)

    rectangle = mpl.patches.Rectangle(xy, width, height, angle=angle, ec=ec, fc=fc, fill=True, linewidth=linewidth,
                                      linestyle=linestyle, zorder=zorder)
    ax = plt.gca()
    ax.add_artist(rectangle)


def figure_to_image(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    # Option 2a: Convert to a NumPy array.
    img = np.fromstring(s, np.uint8).reshape((height, width, 4))
    return img
