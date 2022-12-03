#!/usr/bin/env python2
import json
import os
import logging
import math
from typing import TYPE_CHECKING

# External imports
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Flex vision imports
from flex_vision.utils import imgpy
from flex_vision.utils import geometry
from flex_vision.utils.timer import Timer

from flex_vision.utils.util import save_img, save_fig, figure_to_image
from flex_vision.utils.util import stack_segments, change_brightness
from flex_vision.utils.util import plot_grasp_location, plot_image, plot_features, plot_segments

from flex_vision.detect_truss.filter_segments import filter_segments
from flex_vision.detect_truss.detect_peduncle_2 import detect_peduncle, visualize_skeleton
from flex_vision.detect_truss.detect_tomato import detect_tomato
from flex_vision.detect_truss.segment_image import segment_truss
from flex_vision.detect_truss import settings

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import typing

logger = logging.getLogger(__name__)


class ProcessImage(object):
    # frame ids
    ORIGINAL_FRAME_ID = 'original'
    LOCAL_FRAME_ID = 'local'

    name_space = 'main'  # used for timer

    def __init__(self, save=False, pwd='', name='tomato', ext='pdf'):
        self.ext = ext
        self.save = save
        self.pwd = pwd
        self.name = name

        self.img_rgb = None
        self.shape = None
        self.px_per_mm = None

        # color space
        self.img_a = None
        self.img_hue = None

        # segments
        self.background = None
        self.tomato = None
        self.peduncle = None

        # crop
        self.bbox = None
        self.angle = 0.0
        self.transform = geometry.Transform(self.ORIGINAL_FRAME_ID, self.LOCAL_FRAME_ID, 0.0, 0.0, 0.0)  # identity

        self.truss_crop = None
        self.tomato_crop = None
        self.peduncle_crop = None
        self.img_rgb_crop = None

        # tomatoes
        self.centers = []  # type: typing.List[geometry.Point2D]
        self.radii = []    # type: typing.List[float]
        self.com = None   # type: typing.Optional[geometry.Point2D]

        # stem
        self.junction_points = []  # type: typing.List[geometry.Point2D]
        self.end_points = []  # type: typing.List[geometry.Point2D]
        self.peduncle_points = []  # type: typing.List[geometry.Point2D]
        self.branch_data = None

        # grasp location
        self.grasp_point = None
        self.grasp_angle_local = None
        self.grasp_angle_global = None

        self.settings = settings.initialize_all()

    def add_image(self, img_rgb, px_per_mm=None, name=None):

        # TODO: scaling is currently not supported, would be interesting to reduce computing power
        self.img_rgb = img_rgb
        self.shape = img_rgb.shape[:2]
        self.px_per_mm = px_per_mm

        self.grasp_point = None
        self.grasp_angle_local = None
        self.grasp_angle_global = None

        if name is not None:
            self.name = name

    @Timer("color space", name_space)
    def color_space(self):
        pwd = os.path.join(self.pwd, '01_color_space')
        self.img_hue = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2HSV)[:, :, 0]
        self.img_a = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2LAB)[:, :, 1]  # L: 0 to 255, a: 0 to 255, b: 0 to 255

        if self.save:
            save_img(self.img_hue, pwd, self.name + '_h_raw', vmin=0, vmax=180)
            save_img(self.img_a, pwd, self.name + '_a_raw', vmin=1, vmax=255)

            save_img(self.img_hue, pwd, self.name + '_h', color_map='HSV', vmin=0, vmax=180)
            save_img(self.img_a, pwd, self.name + '_a', color_map='Lab')

    @Timer("segmentation", name_space)
    def segment_image(self, radius=None):
        """segment image based on hue and a color components.

        Keyword arguments:
        radius -- hue circle radius (see https://surfdrive.surf.nl/files/index.php/s/StoH7xA87zUxl79 page 16)
        """

        if radius is None:
            pwd = os.path.join(self.pwd, '02_segment')
        else:
            pwd = os.path.join(self.pwd, '02_segment', str(radius))
            self.settings['segment_image']['hue_radius'] = radius

        success = True
        background, tomato, peduncle = segment_truss(self.img_hue,
                                                     img_a=self.img_a,
                                                     my_settings=self.settings['segment_image'],
                                                     save=self.save,
                                                     pwd=pwd,
                                                     name=self.name)

        self.background = background
        self.tomato = tomato
        self.peduncle = peduncle

        if not self.tomato.any():
            logger.warning("Failed to segment image: no pixel has been classified as tomato")
            success = False

        if not self.peduncle.any():
            logger.warning("Failed to segment image: no pixel has been classified as peduncle")
            success = False

        if self.save:
            self.save_results(self.name, pwd=pwd)

        logger.debug("Successfully segmented the image into %.1f%% tomato, %.1f%% peduncle and %.1f%% background",
                     100 * float(np.count_nonzero(self.tomato)) / self.img_hue.size,
                     100 * float(np.count_nonzero(self.peduncle)) / self.img_hue.size,
                     100 * float(np.count_nonzero(self.background)) / self.img_hue.size)
        return success

    @Timer("filtering", name_space)
    def filter_image(self, folder_name=None):
        """ remove small pixel blobs from the determined segments

        Keyword arguments:
        folder_name -- name of folder where results are stored
        """
        pwd = os.path.join(self.pwd, '03_filter')
        if folder_name is not None:
            pwd = os.path.join(pwd, folder_name)

        # Store member locally such that they can be compared
        tomato = self.tomato
        peduncle = self.peduncle
        background = self.background

        self.tomato, self.peduncle, self.background = filter_segments(tomato,
                                                                      peduncle,
                                                                      background,
                                                                      settings=self.settings['filter_segments'])

        if self.save:
            self.save_results(self.name, pwd=pwd)

        if not np.count_nonzero(self.tomato):
            logger.warning("Failed to filter segments: no pixel has been classified as tomato")
            return False

        if not np.count_nonzero(self.peduncle):
            logger.warning("Failed to filter segments: no pixel has been classified as stem")
            return False

        logger.debug("Successfully filtered segments into %.1f%% tomato, %.1f%% peduncle and %.1f%% background",
                     100 * float(np.count_nonzero(self.tomato)) / self.tomato.size,
                     100 * float(np.count_nonzero(self.peduncle)) / self.tomato.size,
                     100 * float(np.count_nonzero(self.background)) / self.tomato.size)

        return True

    @Timer("cropping", name_space)
    def rotate_cut_img(self):
        """crop the image"""
        pwd = os.path.join(self.pwd, '04_crop')

        if not self.peduncle.sum():
            logger.warning("Cannot rotate based on peduncle since it is empty")
            angle = 0.0
        else:
            angle = imgpy.compute_angle(self.peduncle)  # [rad]

        tomato_rotate = imgpy.rotate(self.tomato, -angle)
        peduncle_rotate = imgpy.rotate(self.peduncle, -angle)
        truss_rotate = imgpy.add(tomato_rotate, peduncle_rotate)

        if not truss_rotate.sum():
            logger.warning("Cannot crop based on truss segment since it is empty")
            return False

        self.bbox = imgpy.bbox(truss_rotate)

        self.transform = geometry.image_transform(
            self.ORIGINAL_FRAME_ID,
            self.LOCAL_FRAME_ID,
            self.shape,
            angle=-angle,
            tx=self.bbox[0],
            ty=self.bbox[1]
        )

        self.angle = angle
        self.tomato_crop = imgpy.cut(tomato_rotate, self.bbox)
        self.peduncle_crop = imgpy.cut(peduncle_rotate, self.bbox)
        self.img_rgb_crop = imgpy.crop(self.img_rgb, angle=-angle, bounding_box=self.bbox)
        self.truss_crop = imgpy.cut(truss_rotate, self.bbox)
        logger.debug("Successfully cropped image")

        if self.save:
            img_rgb = self.get_rgb(local=True)
            save_img(img_rgb, pwd=pwd, name=self.name)
            # self.save_results(self.name, pwd=pwd, local=True)

    @Timer("tomato detection", name_space)
    def detect_tomatoes(self):
        """detect tomatoes based on the truss segment"""
        pwd = os.path.join(self.pwd, '05_tomatoes')

        if not self.truss_crop.sum():
            logger.warning("Cannot detect tomatoes: no pixel has been classified as truss")
            return False

        centers, radii, com = detect_tomato(self.truss_crop,
                                            self.settings['detect_tomato'],
                                            px_per_mm=self.px_per_mm,
                                            img_rgb=self.img_rgb_crop,
                                            save=self.save,
                                            pwd=pwd,
                                            name=self.name)

        # convert obtained coordinated to two-dimensional points linked to a coordinate frame
        self.radii = radii
        self.centers = geometry.points_from_coords(centers, self.LOCAL_FRAME_ID)
        self.com = geometry.Point2D(com, self.LOCAL_FRAME_ID)

        if centers is None or not centers.any():
            logger.warning("Failed to detect any tomatoes in image %s", self.name)
            return False
        else:
            logger.debug("Successfully detected %d tomatoes in image %s", len(radii), self.name)
            return True

    @Timer("peduncle detection", name_space)
    def detect_peduncle(self):
        """Detect the peduncle and junctions"""
        pwd = os.path.join(self.pwd, '06_peduncle')
        if self.save:
            img_bg = change_brightness(self.get_segmented_image(local=True), 0.85)
        else:
            img_bg = self.img_rgb_crop

        mask, branch_data, junc_coords, end_coords = detect_peduncle(self.peduncle_crop,
                                                                     self.settings['detect_peduncle'],
                                                                     px_per_mm=self.px_per_mm,
                                                                     save=self.save,
                                                                     bg_img=img_bg,
                                                                     name=self.name,
                                                                     pwd=pwd)
        # convert to 2D points
        self.peduncle_points = geometry.points_from_coords(np.argwhere(mask)[:, (1, 0)], self.LOCAL_FRAME_ID)
        self.junction_points = geometry.points_from_coords(junc_coords, self.LOCAL_FRAME_ID)
        self.end_points = geometry.points_from_coords(end_coords, self.LOCAL_FRAME_ID)

        # extract branch data
        for branch_type in branch_data:
            for i, branch in enumerate(branch_data[branch_type]):
                for lbl in ['coords', 'src_node_coord', 'dst_node_coord', 'center_node_coord']:

                    if lbl == 'coords':
                        branch_data[branch_type][i][lbl] = geometry.points_from_coords(branch[lbl], self.LOCAL_FRAME_ID)
                    else:
                        branch_data[branch_type][i][lbl] = geometry.Point2D(branch[lbl], self.LOCAL_FRAME_ID)

        self.branch_data = branch_data

        # log result depending on whether any peduncle points have been detected
        if self.peduncle_points:
            logger.debug("Successfully detected peduncle consisting of %d points with %d junctions and %d ends",
                         len(self.peduncle_points), len(self.junction_points), len(self.end_points))
            return True
        else:
            logger.warning("Failed to detect any peduncle points")
            return False

    @Timer("detect grasp location", name_space)
    def detect_grasp_location(self):
        """Determine grasp location based on peduncle, junction and com information"""
        pwd = os.path.join(self.pwd, '07_grasp')

        grasp_settings = self.settings['compute_grasp']

        # set dimensions
        if self.px_per_mm is not None:
            minimum_grasp_length_px = self.px_per_mm * grasp_settings['grasp_length_min_mm']
        else:
            minimum_grasp_length_px = grasp_settings['grasp_length_min_px']

        points_keep = []
        branches_i = []
        for branch_i, branch in enumerate(self.branch_data['junction-junction']):
            if branch['length'] > minimum_grasp_length_px:
                src_node_dist = [geometry.distance(branch['src_node_coord'], coord) for coord in branch['coords']]
                dst_node_dist = [geometry.distance(branch['dst_node_coord'], coord) for coord in branch['coords']]
                is_true = np.logical_and(
                    (np.array(dst_node_dist) > 0.5 * minimum_grasp_length_px),
                    (np.array(src_node_dist) > 0.5 * minimum_grasp_length_px)
                )

                branch_points_keep = np.array(branch['coords'])[is_true].tolist()
                points_keep.extend(branch_points_keep)
                branches_i.extend([branch_i] * len(branch_points_keep))

        if not branches_i:
            logger.warning("Failed to detect a valid grasping branch")

            if self.save:
                save_img(self.img_rgb_crop, pwd=pwd, name=self.name)
                save_img(self.img_rgb, pwd=pwd, name=self.name + '_g')

            return False

        # Select optimal grasp location
        i_grasp = np.argmin([geometry.distance(self.com, point_keep) for point_keep in points_keep])
        grasp_point = points_keep[i_grasp]
        branch_i = branches_i[i_grasp]

        grasp_angle_local = math.radians(self.branch_data['junction-junction'][branch_i]['angle'])
        grasp_angle_global = -self.angle + grasp_angle_local

        self.grasp_point = grasp_point
        self.grasp_angle_local = grasp_angle_local
        self.grasp_angle_global = grasp_angle_global
        logger.debug("Successfully determined grasp location at x: %dpx, y: %dpx with angle %.1fdeg",
                     grasp_point.coord[0], grasp_point.coord[1], math.degrees(grasp_angle_local))

        if not self.save:
            return True

        open_dist_px = grasp_settings['open_dist_mm'] * self.px_per_mm
        finger_thickness_px = grasp_settings['finger_thinkness_mm'] * self.px_per_mm
        brightness = 0.85

        for frame_id in [self.LOCAL_FRAME_ID, self.ORIGINAL_FRAME_ID]:
            grasp_coord = self.transform.apply(grasp_point, frame_id).coord

            if frame_id == self.LOCAL_FRAME_ID:
                grasp_angle = self.grasp_angle_local
                img_rgb = self.img_rgb_crop

            elif frame_id == self.ORIGINAL_FRAME_ID:
                grasp_angle = self.grasp_angle_global
                img_rgb = self.img_rgb

            img_rgb_bright = change_brightness(img_rgb, brightness)
            branch_image = np.zeros(img_rgb_bright.shape[0:2], dtype=np.uint8)
            coords = np.rint(geometry.coords_from_points(self.transform, points_keep, frame_id)).astype(np.int)
            branch_image[coords[:, 1], coords[:, 0]] = 255

            if frame_id == self.ORIGINAL_FRAME_ID:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                branch_image = cv2.dilate(branch_image, kernel, iterations=1)

            visualize_skeleton(img_rgb_bright, branch_image, show_nodes=False, skeleton_color=(0, 0, 0),
                               skeleton_width=4)
            plot_grasp_location(grasp_coord, grasp_angle, finger_width=minimum_grasp_length_px,
                                finger_thickness=finger_thickness_px, finger_dist=open_dist_px, pwd=pwd,
                                name=self.name + '_' + frame_id, linewidth=3)

        return True

    def crop(self, image):
        return imgpy.crop(image, angle=-self.angle, bounding_box=self.bbox)

    def get_tomatoes(self, local=False):
        target_frame_id = self.LOCAL_FRAME_ID if local else self.ORIGINAL_FRAME_ID
        if self.centers is None:
            radii = []
            xy_centers = [[]]
            xy_com = []
            row = []
            col = []

        else:
            xy_centers = geometry.coords_from_points(self.transform, self.centers, target_frame_id)
            xy_com = self.transform.apply(self.com, target_frame_id).coord
            radii = self.radii.tolist()

            row = [xy_center[1] for xy_center in xy_centers]
            col = [xy_center[0] for xy_center in xy_centers]

        tomato = {'centers': xy_centers, 'radii': radii, 'com': xy_com, "row": row, "col": col}
        return tomato

    def get_peduncle(self, local=False):
        """Returns a dictionary containing a description of the peduncle"""

        if local:
            frame_id = self.LOCAL_FRAME_ID
        else:
            frame_id = self.ORIGINAL_FRAME_ID

        peduncle_xy = geometry.coords_from_points(self.transform, self.peduncle_points, frame_id)
        junc_xy = geometry.coords_from_points(self.transform, self.junction_points, frame_id)
        end_xy = geometry.coords_from_points(self.transform, self.end_points, frame_id)
        peduncle = {'junctions': junc_xy, 'ends': end_xy, 'peduncle': peduncle_xy}
        return peduncle

    def get_grasp_location(self, local=False):
        """Returns a dictionary containing a description of the peduncle"""
        if local:
            frame_id = self.LOCAL_FRAME_ID
            angle = self.grasp_angle_local
        else:
            frame_id = self.ORIGINAL_FRAME_ID
            angle = self.grasp_angle_global
        if self.grasp_point is not None:
            xy = self.transform.apply(self.grasp_point, frame_id).coord
            grasp_pixel = np.around(xy).astype(int)
            row = grasp_pixel[1]
            col = grasp_pixel[0]
        else:
            row = None
            col = None
            xy = []

        grasp_location = {"xy": xy, "row": row, "col": col, "angle": angle}
        return grasp_location

    def get_object_features(self):
        """
        Returns a dictionary containing the grasp_location, peduncle, and tomato
        """
        tomatoes = self.get_tomatoes()
        peduncle = self.get_peduncle()
        grasp_location = self.get_grasp_location()

        object_feature = {
            "grasp_location": grasp_location,
            "tomato": tomatoes,
            "peduncle": peduncle
        }
        return object_feature

    def get_tomato_visualization(self, local=False):
        if local is True:
            frame = self.LOCAL_FRAME_ID
            zoom = True
        else:
            frame = self.ORIGINAL_FRAME_ID
            zoom = False

        img_rgb = self.get_rgb(local=local)
        centers = geometry.coords_from_points(self.transform, self.centers, frame)
        com = self.transform.apply(self.com, frame).coord

        tomato = {'centers': centers, 'radii': self.radii, 'com': com}
        plot_features(img_rgb, tomato=tomato, zoom=zoom)
        return figure_to_image(plt.gcf())

    def get_rgb(self, local=False):
        if local:
            return self.img_rgb_crop
        else:
            return self.img_rgb

    def get_truss_visualization(self, local=False, save=False):
        pwd = os.path.join(self.pwd, '08_result')

        if local:
            frame_id = self.LOCAL_FRAME_ID
            shape = self.shape  # self.bbox[2:4]
            zoom = True
            name = 'local'
            skeleton_width = 4
            grasp_linewidth = 3
        else:
            frame_id = self.ORIGINAL_FRAME_ID
            shape = self.shape
            zoom = False
            name = 'original'
            skeleton_width = 2
            grasp_linewidth = 1

        grasp = self.get_grasp_location(local=local)
        tomato = self.get_tomatoes(local=local)
        xy_junc = geometry.coords_from_points(self.transform, self.junction_points, frame_id)
        img = self.get_rgb(local=local)

        # generate peduncle image
        xy_peduncle = geometry.coords_from_points(self.transform, self.peduncle_points, frame_id)
        rc_peduncle = np.around(np.array(xy_peduncle)).astype(np.int)[:, (1, 0)]
        arr = np.zeros(shape, dtype=np.uint8)
        arr[rc_peduncle[:, 0], rc_peduncle[:, 1]] = 1

        # plot
        plt.figure()
        plot_image(img)
        plot_features(tomato=tomato, zoom=zoom)
        visualize_skeleton(img, arr, coord_junc=xy_junc, show_img=False, skeleton_width=skeleton_width)

        if (grasp["xy"] is not None) and (grasp["angle"] is not None):
            grasp_settings = self.settings['compute_grasp']
            if self.px_per_mm is not None:
                minimum_grasp_length_px = self.px_per_mm * grasp_settings['grasp_length_min_mm']
                open_dist_px = grasp_settings['open_dist_mm'] * self.px_per_mm
                finger_thickenss_px = grasp_settings['finger_thinkness_mm'] * self.px_per_mm
            else:
                minimum_grasp_length_px = grasp_settings['grasp_length_min_px']
            plot_grasp_location(grasp["xy"], grasp["angle"], finger_width=minimum_grasp_length_px,
                                finger_thickness=finger_thickenss_px, finger_dist=open_dist_px, linewidth=grasp_linewidth)

        if save:
            if name is None:
                name = self.name
            else:
                name = self.name + '_' + name
            save_fig(plt.gcf(), pwd, name)

        return figure_to_image(plt.gcf())

    def get_segments(self, local=False):
        if local:
            tomato = self.tomato_crop  # self.crop(self.tomato)
            peduncle = self.peduncle_crop  # self.crop(self.peduncle)
            background = self.crop(self.background)
        else:
            tomato = self.tomato
            peduncle = self.peduncle
            background = self.background

        return tomato, peduncle, background

    def get_segmented_image(self, local=False):
        tomato, peduncle, background = self.get_segments(local=local)
        image_rgb = self.get_rgb(local=local)
        data = stack_segments(image_rgb, background, tomato, peduncle)
        return data

    def get_color_components(self):
        return self.img_hue

    def save_results(self, name, local=False, pwd=None):
        if pwd is None:
            pwd = self.pwd

        tomato, peduncle, background = self.get_segments(local=local)
        img_rgb = self.get_rgb(local=local)
        plot_segments(img_rgb, background, tomato, peduncle, linewidth=0.5, pwd=pwd, name=name, alpha=1)

    @Timer("process image")
    def process_image(self):
        """
        Apply entire image processing pipeline
        returns: True if success, False if failed
        """

        self.color_space()

        success = self.segment_image()
        if success is False:
            logger.warning("Failed to process image: failed to segment image")
            return success

        success = self.filter_image()
        if success is False:
            logger.warning("Failed to process image: failed to filter image")
            return success

        success = self.rotate_cut_img()
        if success is False:
            logger.warning("Failed to process image: failed to crop image")
            return success

        success = self.detect_tomatoes()
        if success is False:
            logger.warning("Failed to process image: failed to detect tomatoes")
            return success

        success = self.detect_peduncle()
        if success is False:
            logger.warning("Failed to process image: failed to detect peduncle")
            return success

        success = self.detect_grasp_location()
        if success is False:
            logger.warning("Failed to process image: failed to detect grasp location")
        return success

    def get_settings(self):
        return self.settings

    def set_settings(self, my_settings):
        """
        Overwrites the settings which are present in the given dict
        """

        for key_1 in my_settings:
            for key_2 in my_settings[key_1]:
                self.settings[key_1][key_2] = my_settings[key_1][key_2]


def load_px_per_mm(pwd, img_id):
    pwd_info = os.path.join(pwd, img_id + '_info.json')

    if not os.path.exists(pwd_info):
        print('Info does not exist for image: ' + img_id + ' continuing without info')
        return None

    with open(pwd_info, "r") as read_file:
        data_inf = json.load(read_file)

    return data_inf['px_per_mm']
