#!/usr/bin/env python2
import argparse
import os
import json

# External imports
import numpy as np
import matplotlib.pyplot as plt

# Flex vision imports
from flex_vision import constants
from flex_vision.detect_truss.ProcessImage import ProcessImage, load_px_per_mm
from flex_vision.utils import util
from flex_vision.utils.timer import Timer


def main():
    parser = argparse.ArgumentParser(prog="process_image.py", description="Process images")
    parser.add_argument("--plot_timer", help="Plot the timer after processing", action="store_true")

    args = parser.parse_args()
    plot_timer = args.plot_timer

    i_start = 3
    i_end = 4
    N = i_end - i_start

    save = False
    pwd_data = constants.PATH_DATA
    pwd_results = constants.PATH_RESULTS
    pwd_json = constants.PATH_JSON

    util.make_dirs(pwd_results)
    util.make_dirs(pwd_json)

    process_image = ProcessImage(use_truss=True,
                                 pwd=pwd_results,
                                 save=save)

    for count, i_tomato in enumerate(range(i_start, i_end)):
        print("Analyzing image ID %d (%d/%d)" % (i_tomato, count + 1, N))

        tomato_name = str(i_tomato).zfill(3)
        file_name = tomato_name + ".png"

        rgb_data = util.load_rgb(file_name, pwd=pwd_data, horizontal=True)
        px_per_mm = load_px_per_mm(pwd_data, tomato_name)
        process_image.add_image(rgb_data, px_per_mm=px_per_mm, name=tomato_name)

        success = process_image.process_image()
        process_image.get_truss_visualization(local=True, save=True)
        process_image.get_truss_visualization(local=False, save=True)

        json_data = process_image.get_object_features()

        pwd_json_file = os.path.join(pwd_json, tomato_name + '.json')
        with open(pwd_json_file, "w") as write_file:
            json.dump(json_data, write_file)

    if plot_timer:
        util.plot_timer(Timer.timers['main'].copy(), threshold=0.02, pwd=pwd_results, name='main', title='Processing time',
                        startangle=-20)

    total_pixels = process_image.background_pixels + process_image.tomato_pixels + process_image.stem_pixels
    print(float(process_image.background_pixels)/total_pixels)
    print(float(process_image.tomato_pixels) / total_pixels)
    print(float(process_image.stem_pixels) / total_pixels)

    total_key = "process image"
    time_tot_mean = np.mean(Timer.timers[total_key]) / 1000
    time_tot_std = np.std(Timer.timers[total_key]) / 1000

    time_ms = Timer.timers[total_key]
    time_s = [x / 1000 for x in time_ms]

    time_min = min(time_s)
    time_max = max(time_s)

    print 'Processing time: {mean:.2f}s +- {std:.2f}s (n = {n:d})'.format(mean=time_tot_mean, std=time_tot_std, n=N)
    print 'Processing time lies between {time_min:.2f}s and {time_max:.2f}s (n = {n:d})'.format(time_min=time_min,
                                                                                                time_max=time_max, n=N)

    width = 0.5
    fig, ax = plt.subplots()

    ax.p1 = plt.bar(np.arange(i_start, i_end), time_s, width)

    plt.ylabel('time [s]')
    plt.xlabel('image ID')
    plt.title('Processing time per image')
    plt.rcParams["savefig.format"] = 'pdf'

    fig.show()
    fig.savefig(os.path.join(pwd_results, 'time_bar'), dpi=300)  # , bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()
