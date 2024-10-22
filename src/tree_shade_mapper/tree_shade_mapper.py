import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os

from multiprocessing import Pool, cpu_count
from functools import partial

from .image_process import segmentation_dir, resize_dir
from .image_process import get_transmittance_center_of_modes_upper_v2#, get_transmittance_seg, get_transmittance_bin
from .image_process import get_sky_view_factor_from_binary

from .solar_data_process import create_solar_time_series, create_solar_instantaneous 
from .solar_data_process import calc_solar_irradiance_under_tree_map, calc_solar_irradiance_under_tree_spontaneous
from .solar_data_process import interval_to_seconds
from .visualization import mapping_accu, mapping_time_series, mapping_svf, create_video

binary_type = "brightness"

def calc_transmittance(base_dir, models=["tcm"], image_size = (2048, 1024), calc_type = None):

    kernel_size = int(image_size[0]/64)

    #segmentation
    ori_dir = f"{base_dir}/original"
    res_dir = f"{base_dir}/resized"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    resize_dir(ori_dir, res_dir, new_size = image_size)
    seg_dir = f"{base_dir}/segmented"
    segmentation_dir(res_dir, seg_dir)

    if calc_type == None:
        locations = [pano.replace(".jpg", "") for pano in os.listdir(res_dir)]

        df_svf_list = []
        for model in models:
            #prepare directories for results
            tra_dir = f"{base_dir}/transmittance_{model}"
            if not os.path.exists(tra_dir):
                os.makedirs(tra_dir)
            bin_dir = f"{base_dir}/binary_{model}"
            if not os.path.exists(bin_dir):
                os.makedirs(bin_dir)

            svf_list = []
            for location in locations:
                ori_path = os.path.join(res_dir, location+".jpg")
                seg_path = os.path.join(seg_dir, location+"_colored_segmented.png")
                tra_path = os.path.join(tra_dir, location+"_tra.npy")
                bin_path = os.path.join(bin_dir, location+"_bin.npy")
                tra_img_path = os.path.join(tra_dir, location+"_tra.jpg")
                bin_img_path = os.path.join(bin_dir, location+"_bin.jpg")
                svf_img_path = os.path.join(bin_dir, location+"_svf.jpg")
                
                if os.path.exists(tra_path) & os.path.exists(bin_path):
                    array_binary = np.load(bin_path)
                    array_transmittance = np.load(tra_path)
                else:
                    array_transmittance, array_binary = get_transmittance_center_of_modes_upper_v2(ori_path, seg_path, kernel_size, binary_type = binary_type, model = model, type = 'q2')
                    np.save(bin_path, array_binary)
                    np.save(tra_path, array_transmittance)
                # magma_array_transmittance = mcm.magma(array_transmittance, bytes=True)
                # tra_img = Image.fromarray(magma_array_transmittance * 255, 'RGB')
                # tra_img.save(tra_img_path)
                plt.imsave(tra_img_path, array_transmittance, cmap='magma')
                
                # bin_img = Image.fromarray(array_binary * 255, 'L')
                # bin_img.save(bin_img_path)
                # display_image(array_binary, 'gray')
                # display_image(array_transmittance, 'magma')
                plt.imsave(bin_img_path, array_binary, cmap='gray')

                # display_image(array_seg_binary, 'gray')
                # display_image(array_transmittance, 'jet')

                sky_view_factor, svf_binary = get_sky_view_factor_from_binary(array_binary)
                svf_list.append([location, sky_view_factor])
                plt.imsave(svf_img_path, svf_binary, cmap='gray')

            # df_frames[f"svf_{model}"] = svf_list
            df_svf = pd.DataFrame(svf_list, columns=['location', f'svf_{model}'])
            df_svf.set_index('location', inplace=True)
            df_svf_list.append(df_svf)

        df_svf_list
        df_frames = pd.concat(df_svf_list, axis=1, ignore_index=False)

    elif calc_type == 'map':
        df_path = f"{base_dir}/frames.csv"
        df_frames = pd.read_csv(df_path)

        for model in models:
            #prepare directories for results
            tra_dir = f"{base_dir}/transmittance_{model}"
            if not os.path.exists(tra_dir):
                os.makedirs(tra_dir)
            bin_dir = f"{base_dir}/binary_{model}"
            if not os.path.exists(bin_dir):
                os.makedirs(bin_dir)

            svf_list = []
            for index, row in df_frames.iterrows():
                ori_path = os.path.join(res_dir, row["frame_key"]+".jpg")
                seg_path = os.path.join(seg_dir, row["frame_key"]+"_colored_segmented.png")
                tra_path = os.path.join(tra_dir, row["frame_key"]+"_tra.npy")
                bin_path = os.path.join(bin_dir, row["frame_key"]+"_bin.npy")
                tra_img_path = os.path.join(tra_dir, row["frame_key"]+"_tra.jpg")
                bin_img_path = os.path.join(bin_dir, row["frame_key"]+"_bin.jpg")
                svf_img_path = os.path.join(bin_dir, row["frame_key"]+"_svf.jpg")
                
                if os.path.exists(tra_path) & os.path.exists(bin_path):
                    array_binary = np.load(bin_path)
                    array_transmittance = np.load(tra_path)
                else:
                    array_transmittance, array_binary = array_transmittance, array_binary = get_transmittance_center_of_modes_upper_v2(ori_path, seg_path, kernel_size, binary_type = binary_type, model = model, type = "q2")
                    np.save(tra_path, array_transmittance)
                    np.save(bin_path, array_binary)

                plt.imsave(tra_img_path, array_transmittance, cmap='magma')
                plt.imsave(bin_img_path, array_binary, cmap='gray')

                sky_view_factor, svf_binary = get_sky_view_factor_from_binary(array_binary)
                svf_list.append(sky_view_factor)
                plt.imsave(svf_img_path, svf_binary, cmap='gray')

            df_frames[f"svf_{model}"] = svf_list

    df_svf_path = f"{base_dir}/frames_svf.csv"
    df_frames.to_csv(df_svf_path)

def calc_solar_irradiance(base_dir, time_start, time_end, interval, time_zone, latitude, longitude, azimuth_offset=180, models=["tcm"]):

    df_svf_path = f"{base_dir}/frames_svf.csv"
    df_frames = pd.read_csv(df_svf_path)

    df_solar_ori = create_solar_time_series(time_start, time_end, time_zone, interval, latitude, longitude)

    df_solar_list = []
    for index, row in df_frames.iterrows():
        df_solar = df_solar_ori.copy()

        for model in models:
            tra_dir = f"{base_dir}/transmittance_{model}"
            tra_path = os.path.join(tra_dir, row["frame_key"]+"_tra.npy")    
            array_transmittance = np.load(tra_path)
            # calc_solar_irradiance_under_tree_v2(df_solar, array_transmittance, row["svf"], azimuth_offset = azimuth_offset)
            calc_solar_irradiance_under_tree_map(df_solar, array_transmittance, row[f"svf_{model}"], azimuth_offset = azimuth_offset, model = model)

        df_solar["frame_key"] = row["frame_key"]
        df_solar["lat"] = row["lat"]
        df_solar["lon"] = row["lon"]
        # df_solar["precision"] = row["precision"]
        df_solar_list.append(df_solar)

    # Concatenate all dataframes in the list
    df_solar_all = pd.concat(df_solar_list, axis=0, ignore_index=False)

    df_solar_path = f"{base_dir}/frames_solar.csv"
    df_solar_all.to_csv(df_solar_path)

    aggregations = {
        'ghi': 'mean',
        'dni': 'mean',
        'dhi': 'mean',
        'apparent_zenith': 'mean',
        'zenith': 'mean',
        'apparent_elevation': 'mean',
        'elevation': 'mean',
        'azimuth': 'mean',
        'equation_of_time': 'mean',
        ## Add all 'ghi_***' like columns that you want to sum
        # 'ghi_utc_segbin': 'sum',
        # 'transmittance_segbin': 'mean',
        # 'ghi_utc_seg': 'sum',
        # 'transmittance_seg': 'mean',
        # 'ghi_utc_bin': 'sum',
        # 'transmittance_bin': 'mean',
        ## Average the latitude and longitude
        'lat': 'mean',
        'lon': 'mean',
        # 'precision': 'mean',
    }

    for model in models:
        aggregations[f'ghi_utc_{model}'] = 'sum'
        aggregations[f'transmittance_{model}'] = 'mean'

    # Group by 'frame_key' and apply the aggregation functions
    df_solar_accu = df_solar_all.groupby('frame_key').agg(aggregations)
    for model in models:
        df_solar_accu[f'ghi_utc_acc_{model}'] = df_solar_accu[f'ghi_utc_{model}'] * interval_to_seconds(interval) / 1000000


    df_solar_accu_path = f"{base_dir}/frames_solar_accu.csv"
    df_solar_accu.to_csv(df_solar_accu_path)

def calc_solar_irradiance_instantaneous(base_dir, target_time, time_zone, latitude, longitude, ghi, azimuth_offset=180):

    model="tcm"

    df_svf_path = f"{base_dir}/svf.csv"
    df_svf = pd.read_csv(df_svf_path)

    df_solar = create_solar_instantaneous(target_time, time_zone, latitude, longitude, ghi, altitude=0, rad_model='erbs_driesse')

    ghi_utc_list = []
    transmittance_list = []
    for index, row in df_svf.iterrows():
        tra_dir = f"{base_dir}/transmittance_{model}"
        tra_path = os.path.join(tra_dir, row["frame_key"]+"_tra.npy")    
        array_transmittance = np.load(tra_path)
        ghi_utc, transmittance = calc_solar_irradiance_under_tree_spontaneous(df_solar, array_transmittance, row[f"svf_{model}"], azimuth_offset = azimuth_offset, model = model)
        ghi_utc_list.append(ghi_utc)
        transmittance_list.append(transmittance)

    df_svf[f"ghi_utc_{model}"] = ghi_utc_list
    # df_svf[f"transmittance_{model}"] = transmittance_list
    # df_svf["zenith"] = df_solar["zenith"].iloc[0]
    # df_svf["dni"] = df_solar["dni"].iloc[0]
    # df_svf["dhi"] = df_solar["dhi"].iloc[0]

    return df_svf

def get_tree_shade(base_dir, time_start, time_end, interval, time_zone, latitude, longitude, models=['tcm'], image_size = (2048, 1024), calc_type=None, vmin=0, vmax=1000, resolution=14):
    calc_transmittance(base_dir, models=models, image_size = image_size, calc_type=calc_type)
    calc_solar_irradiance(base_dir, time_start, time_end, interval, time_zone, latitude, longitude, models=models)

    if calc_type == 'map':
        mapping_accu(base_dir, vmin=vmin, vmax = vmax, models = models, resolution = resolution)
        mapping_time_series(base_dir, models = models, resolution = resolution)
        mapping_svf(base_dir, models = models, resolution = resolution)
        frame_dir = f"resized"
        create_video(base_dir, frame_dir)