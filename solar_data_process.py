import os
import pandas as pd
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3
import pytz
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def read_weather_data(filepath_target, filepath_roof, filepath_reference, latitude, longitude):
    df_ws = pd.read_csv(filepath_target)
    df_ws["time"] = pd.to_datetime(df_ws["Date Time"])
    df_ws.rename(columns={'Solar radiation (W/m²)': 'solar_radiation'}, inplace=True)
    df_ws = df_ws[["time", "solar_radiation"]]
    df_ws = df_ws.resample('T', on='time').mean()
    df_ws

    df_rt = pd.read_excel(filepath_reference)
    df_rt['time'] = pd.to_datetime(df_rt['Date'] + ' ' + df_rt['Time'])
    df_rt["time"] = pd.to_datetime(df_rt["time"])
    #df_rt = df_rt[["time", "Solar radiation (W/m²)"]]
    df_rt.rename(columns={'Solar Radiation, W/m^2, #14 - PGP block 2 rooftop': 'solar_radiation'}, inplace=True)
    df_rt = df_rt[["time", "solar_radiation"]]
    df_rt = df_rt.resample('T', on='time').mean()
    df_rt

    df_join = pd.merge(df_ws, df_rt, on='time', how='inner', suffixes=('_ws', '_rt')).dropna()
    df_join.index = df_join.index.tz_localize('Asia/Singapore')
    df_join

    tus = Location(latitude, longitude, pytz.timezone('Asia/Singapore'))
    irradiance = tus.get_clearsky(df_join.index)
    solar_position = solarposition.get_solarposition(df_join.index, latitude, longitude)
    df_concat = pd.concat([df_join, irradiance, solar_position], axis=1)

    return df_concat

# def read_weather_data(filepath_target, filepath_reference, latitude, longitude):
#     df_ws = pd.read_csv(filepath_target)
#     df_ws["time"] = pd.to_datetime(df_ws["Date Time"])
#     df_ws.rename(columns={'Solar radiation (W/m²)': 'solar_radiation'}, inplace=True)
#     df_ws = df_ws[["time", "solar_radiation"]]
#     df_ws = df_ws.resample('T', on='time').mean()
#     df_ws

#     df_rt = pd.read_excel(filepath_reference)
#     df_rt['time'] = pd.to_datetime(df_rt['Date'] + ' ' + df_rt['Time'])
#     df_rt["time"] = pd.to_datetime(df_rt["time"])
#     #df_rt = df_rt[["time", "Solar radiation (W/m²)"]]
#     df_rt.rename(columns={'Solar Radiation, W/m^2, #14 - PGP block 2 rooftop': 'solar_radiation'}, inplace=True)
#     df_rt = df_rt[["time", "solar_radiation"]]
#     df_rt = df_rt.resample('T', on='time').mean()
#     df_rt

#     df_join = pd.merge(df_ws, df_rt, on='time', how='inner', suffixes=('_ws', '_rt')).dropna()
#     df_join.index = df_join.index.tz_localize('Asia/Singapore')
#     df_join

#     tus = Location(latitude, longitude, pytz.timezone('Asia/Singapore'))
#     irradiance = tus.get_clearsky(df_join.index)
#     solar_position = solarposition.get_solarposition(df_join.index, latitude, longitude)
#     df_concat = pd.concat([df_join, irradiance, solar_position], axis=1)

#     return df_concat

def create_solar_time_series(time_start, time_end, time_zone, interval, latitude, longitude, altitude=0):

    # Generate the datetime range
    time_range = pd.date_range(start=time_start, end=time_end, freq=interval, tz=time_zone)

    # Create the DataFrame
    df_time = pd.DataFrame(time_range, columns=['time'])

    # Set the 'time' column as the index to create a DateTimeIndex
    df_time.set_index('time', inplace=True)

    tus = Location(latitude, longitude, pytz.timezone(time_zone))
    irradiance = tus.get_clearsky(df_time.index)
    solar_position = solarposition.get_solarposition(df_time.index, latitude, longitude, altitude)
    df_solar = pd.concat([df_time, irradiance, solar_position], axis=1)

    return df_solar

def create_solar_time_series_walk(df_walk, time_start, interval, time_zone):

    steps = len(df_walk)
    frequency = f'{interval}L' #msec

    # Generate the datetime range
    time_range = pd.date_range(start=time_start, periods=steps, freq=frequency, tz=time_zone)

    # Create the DataFrame
    df_walk["time"] = time_range

    # Set the 'time' column as the index to create a DateTimeIndex
    df_walk.set_index('time', inplace=True)

    # Assuming time_zone and other necessary parameters are defined appropriately
    df_solar_list = []
    for i, row in df_walk.iterrows():
        # Create a location object for solar calculations
        tus = Location(row['lat'], row['lon'], pytz.timezone(time_zone))
        
        # Create a DatetimeIndex from the single timestamp
        time_index = pd.DatetimeIndex([i])
        
        # Get clearsky data for the current index (datetime)
        irradiance = tus.get_clearsky(time_index)
        
        # Get solar position for the current index (datetime)
        solar_position = solarposition.get_solarposition(time_index, row['lat'], row['lon'])

        df_solar = pd.concat([irradiance, solar_position], axis=1)
        
        df_solar_list.append(df_solar)

    df_solar_all = pd.concat(df_solar_list, axis=0, ignore_index=False)
    df_solar_walk = pd.concat([df_walk, df_solar_all], axis=1)
    df_solar_walk

    return df_solar_walk


def add_erbs_direct_diffuse(df):
    erbs = pvlib.irradiance.erbs(df['solar_radiation_rt'], df["zenith"], df.index)
    df['direct_normal_erbs'] = erbs["dni"]
    df['sky_diffuse_erbs'] = erbs["dhi"]        
    df['sky_diffuse_erbs'] = df.apply(modify_sky_diffuse_erbs, axis=1)
    df['direct_normal_erbs'] = df.apply(modify_direct_normal_erbs, axis=1)

def add_erbs_direct_diffuse(df, roofs):
    erbs = pvlib.irradiance.erbs(df['solar_radiation_rt'], df["zenith"], df.index)
    df['direct_normal_erbs'] = erbs["dni"]
    df['sky_diffuse_erbs'] = erbs["dhi"]        
    df['sky_diffuse_erbs'] = df.apply(modify_sky_diffuse_erbs, axis=1)
    df['direct_normal_erbs'] = df.apply(modify_direct_normal_erbs, axis=1)

def modify_sky_diffuse_erbs(row):
    if row['solar_radiation_rt'] < row['ghi']/2:
        return row['solar_radiation_rt']
    else:
        return row['sky_diffuse_erbs']
    
def modify_direct_normal_erbs(row):
    if row['solar_radiation_rt'] < row['ghi']/2:
        return 0
    else:
        return row['direct_normal_erbs']

def calc_solar_irradiance_under_tree(df_solar, array_transmittance, sky_view_factor, azimuth_offset = 180):
    trans_w = len(array_transmittance[0])
    trans_h = len(array_transmittance)

    irradiances = []
    transmittance_list = []

    for j in range(len(df_solar)):
    #for j in range(30,40):
        azimuth = (df_solar["azimuth"][j]+azimuth_offset)%360#degree
        zenith = df_solar["apparent_zenith"][j]#degree

        if zenith < 90:
          trans_index_azimuth = int(round(trans_w * azimuth / 360, 0))-1
          trans_index_zenith = int(round(trans_h / 2 * zenith / 90, 0))-1
          pixel_transmittance = array_transmittance[trans_index_zenith][trans_index_azimuth]
          direct_irradiance = df_solar["direct_normal_erbs"][j] * np.cos((df_solar["apparent_zenith"][j])/360*np.pi) * pixel_transmittance

        else:
          direct_irradiance = 0
          pixel_transmittance = 0

        #diffuse_irradiance = df_concat["dhi"][j] * df_concat["sky_uppr_ortho"][j]
        diffuse_irradiance = df_solar["sky_diffuse_erbs"][j] * sky_view_factor
        irradiance = direct_irradiance + diffuse_irradiance
        #print(time, azimuth, zenith, direct_irradiance)
        irradiances.append(irradiance)
        transmittance_list.append(pixel_transmittance)

    df_solar["gsi_utc"] = irradiances
    df_solar["transmittance"] = transmittance_list

def calc_solar_irradiance_under_tree_map(df_solar, array_transmittance, sky_view_factor, azimuth_offset = 180, model = "segbin"):
    trans_w = len(array_transmittance[0])
    trans_h = len(array_transmittance)

    irradiances = []
    transmittance_list = []

    for j in range(len(df_solar)):
    #for j in range(30,40):
        azimuth = (df_solar["azimuth"][j]+azimuth_offset)%360#degree
        zenith = df_solar["apparent_zenith"][j]#degree

        if zenith < 90:
          trans_index_azimuth = int(round(trans_w * azimuth / 360, 0))-1
          trans_index_zenith = int(round(trans_h / 2 * zenith / 90, 0))-1
          pixel_transmittance = array_transmittance[trans_index_zenith][trans_index_azimuth]
          direct_irradiance = df_solar["dni"][j] * np.cos((df_solar["apparent_zenith"][j])/360*np.pi) * pixel_transmittance

        else:
          direct_irradiance = 0
          pixel_transmittance = 0

        #diffuse_irradiance = df_concat["dhi"][j] * df_concat["sky_uppr_ortho"][j]
        diffuse_irradiance = df_solar["dhi"][j] * sky_view_factor
        irradiance = direct_irradiance + diffuse_irradiance
        #print(time, azimuth, zenith, direct_irradiance)
        irradiances.append(irradiance)
        transmittance_list.append(pixel_transmittance)

    df_solar[f"ghi_utc_{model}"] = irradiances
    df_solar[f"transmittance_{model}"] = transmittance_list

def calc_solar_irradiance_under_tree_validation(df_solar, location, nearest_roof, array_transmittance, sky_view_factor, azimuth_offset = 180, model = "segbin", radmodel = 'erbs'):
    trans_w = len(array_transmittance[0])
    trans_h = len(array_transmittance)

    irradiances = []
    transmittance_list = []

    for j in range(len(df_solar)):
    #for j in range(30,40):
        if df_solar["zenith"][j] > 0:

            azimuth = (df_solar["azimuth"][j]+azimuth_offset)%360#degree
            zenith = df_solar["zenith"][j]#degree

            if zenith < 90:
                trans_index_azimuth = int(round(trans_w * azimuth / 360, 0))-1
                trans_index_zenith = int(round(trans_h / 2 * zenith / 90, 0))-1
                pixel_transmittance = array_transmittance[trans_index_zenith][trans_index_azimuth]
                direct_irradiance = df_solar[f"DNI_rt_{radmodel}_{nearest_roof}"][j] * np.cos((df_solar["zenith"][j])/360*np.pi) * pixel_transmittance

            else:
                direct_irradiance = 0
                pixel_transmittance = 0
        else:
            direct_irradiance = np.nan
            pixel_transmittance = np.nan

        #diffuse_irradiance = df_concat["dhi"][j] * df_concat["sky_uppr_ortho"][j]
        diffuse_irradiance = df_solar[f"SDI_rt_{radmodel}_{nearest_roof}"][j] * sky_view_factor
        irradiance = direct_irradiance + diffuse_irradiance
        #print(time, azimuth, zenith, direct_irradiance)
        irradiances.append(irradiance)
        transmittance_list.append(pixel_transmittance)

    df_solar[f"GHI_utc_{radmodel}_{model}_{location}"] = irradiances
    df_solar[f"transmittance_{model}_{location}"] = transmittance_list

def calc_solar_irradiance_under_tree_walk(df_solar, array_transmittance, sky_view_factor, azimuth_offset = 180):
    trans_w = len(array_transmittance[0])
    trans_h = len(array_transmittance)

    irradiances = []
    transmittance_list = []

    azimuth = (df_solar["azimuth"]+azimuth_offset)%360#degree
    zenith = df_solar["apparent_zenith"]#degree

    if zenith < 90:
        trans_index_azimuth = int(round(trans_w * azimuth / 360, 0))-1
        trans_index_zenith = int(round(trans_h / 2 * zenith / 90, 0))-1
        pixel_transmittance = array_transmittance[trans_index_zenith][trans_index_azimuth]
        direct_irradiance = df_solar["dni"] * np.cos((df_solar["apparent_zenith"])/360*np.pi) * pixel_transmittance

    else:
        direct_irradiance = 0
        pixel_transmittance = 0

    #diffuse_irradiance = df_concat["dhi"][j] * df_concat["sky_uppr_ortho"][j]
    diffuse_irradiance = df_solar["dhi"] * sky_view_factor
    irradiance = direct_irradiance + diffuse_irradiance
    # df_solar["ghi_utc"] = irradiance
    # df_solar["transmittance"] = pixel_transmittance

    return irradiance


def get_error_metrix(df_solar):
    mse = mean_squared_error(df_solar['solar_radiation_ws'], df_solar['gsi_utc'])
    # take the square root of the mean
    rmse = np.sqrt(mse)
    # calculate the square of the difference between the two columns
    mae = mean_absolute_error(df_solar['solar_radiation_ws'], df_solar['gsi_utc'])
    r2 = r2_score(df_solar['solar_radiation_ws'], df_solar['gsi_utc'])
    return {
        "MSE":mse, 
        "RMSE":rmse, 
        "MAR":mae, 
        "R2":r2,
    }

# Function to find the closest location
def find_closest(df_base, df_target, model):
    frames = []
    svfs = []
    for idx, row in df_target.iterrows():
        dist = np.sqrt((df_base['lat'] - row['lat'])**2 + (df_base['lon'] - row['lon'])**2)
        closest_idx = dist.idxmin()
        frames.append(df_base.loc[closest_idx, 'frame_key'])
        svfs.append(df_base.loc[closest_idx, f'svf_{model}'])
    df_target['frame_key'] = frames
    df_target[f'svf_{model}'] = svfs
    return df_target

def find_closest_accu(df_base, df_target, model):
    frames = []
    accughis = []
    for idx, row in df_target.iterrows():
        dist = np.sqrt((df_base['lat'] - row['lat'])**2 + (df_base['lon'] - row['lon'])**2)
        closest_idx = dist.idxmin()
        frames.append(df_base.loc[closest_idx, 'frame_key'])
        accughis.append(df_base.loc[closest_idx, f'ghi_utc_{model}'])
    df_target['frame_key'] = frames
    df_target[f'ghi_utc_{model}'] = accughis
    return df_target

def calc_walk_accumulated_ghi(walk_base_dir, map_base_dir, walk_id, time_start, walk_speed, interval_d, time_zone, azimuth_offset, model="segbin"):

    # set path for directory of transmittance
    tra_dir = f"{map_base_dir}/transmittance_{model}"
    if not os.path.exists(tra_dir):
        os.makedirs(tra_dir)

    # load dataframe for walking path
    df_walk_path = f"{walk_base_dir}/{walk_id}.csv"
    df_walk = pd.read_csv(df_walk_path)

    # time difference between neighboring 2 locations, msec
    interval_t = int(interval_d / walk_speed * 1000)

    # set timestamps and sun positions for each locations
    df_solar_walk = create_solar_time_series_walk(df_walk, time_start, interval_t, time_zone)

    # load dataframe for frame keys and sky view factor
    df_frames = f"{map_base_dir}/frames_svf.csv"
    df_frames = pd.read_csv(df_frames)

    # Find closest frame_key from df_frames for each entry in df_solar_walk
    df_solar_walk_frames = find_closest(df_frames, df_solar_walk, model)    

    # Assuming time_zone and other necessary parameters are defined appropriately
    ghi_utc_list = []
    ghi_utc_accu_list = []
    ghi_utc_accu = 0
    for i, row in df_solar_walk_frames.iterrows():
        tra_path = os.path.join(tra_dir, row["frame_key"]+"_tra.npy")
        array_transmittance = array = np.load(tra_path)
        ghi_utc = calc_solar_irradiance_under_tree_walk(row, array_transmittance, row[f"svf_{model}"], azimuth_offset = azimuth_offset)
        ghi_utc_list.append(ghi_utc)
        ghi_utc_accu += ghi_utc * interval_t / 1000000 #unit KJ/m^2
        ghi_utc_accu_list.append(ghi_utc_accu)
        
    df_solar_walk_frames["ghi_utc"] = ghi_utc_list
    df_solar_walk_frames["ghi_utc_accu"] = ghi_utc_accu_list
    df_solar_walk_frames.reset_index(inplace=True)
    df_solar_walk_frames["distance"] = df_solar_walk_frames.index * interval_d

    return df_solar_walk_frames

def load_walk_accumulated_ghi(walk_base_dir, map_base_dir, walk_id, time_start, walk_speed, interval_d, time_zone, azimuth_offset, model="segbin"):

    # load dataframe for walking path
    df_walk_path = f"{walk_base_dir}/{walk_id}.csv"
    df_walk = pd.read_csv(df_walk_path)

    # time difference between neighboring 2 locations, msec
    # interval_t = int(interval_d / walk_speed * 1000)

    # # set timestamps and sun positions for each locations
    # df_solar_walk = create_solar_time_series_walk(df_walk, time_start, interval_t, time_zone)

    # load dataframe for frame keys and sky view factor
    df_frames = f"{map_base_dir}/frames_solar_accu.csv"
    df_frames = pd.read_csv(df_frames)

    # Find closest frame_key from df_frames for each entry in df_solar_walk
    df_solar_walk_frames = find_closest_accu(df_frames, df_walk, model)   
    
    df_solar_walk_frames.reset_index(inplace=True)
    df_solar_walk_frames["distance"] = df_solar_walk_frames.index * interval_d

    return df_solar_walk_frames

def join_irradiance_data(basedir, monthtag, lat, lon):

    df_locations = pd.read_csv(f'{basedir}/locations.csv')# Create a nested dictionary with 'location' as the key
    location_dict = {row['location']: {'nearest_roof': row['nearest_roof'], 'azimuth_offset': row['azimuth_offset']} for index, row in df_locations.iterrows()}
    tarnames = df_locations['location'].unique().tolist()
    roofnames = df_locations['nearest_roof'].unique().tolist()
    
    time_zone = pytz.timezone('Asia/Singapore')
    
    refnames = [
        "S24_changi"
    ]

    tardir = f"{basedir}/irradiance/target"
    roofdir = f"{basedir}/irradiance/roof"
    refdir = f"{basedir}/irradiance/reference"

    for i, tarname in enumerate(tarnames):
        df_tar = pd.read_csv(f"{tardir}/NUS_{tarname}_{monthtag}.csv")
        df_tar["Datetime"] = pd.to_datetime(df_tar["Datetime"])
        df_tar = df_tar[["Datetime", "GlobalRad Ave (W/m2)"]]
        df_tar = df_tar.rename(columns={"GlobalRad Ave (W/m2)": f"GHI_utc_{tarname}"})
        if i == 0:
            df_tars = df_tar
        else:
            df_tars = pd.merge(df_tars, df_tar, on='Datetime', how='outer')

    for i, roofname in enumerate(roofnames):
        df_roof = pd.read_csv(f"{roofdir}/NUS_{roofname}_{monthtag}.csv")
        df_roof["Datetime"] = pd.to_datetime(df_roof["Datetime"])
        df_roof = df_roof[["Datetime", "GlobalRad Ave (W/m2)"]]
        df_roof = df_roof.rename(columns={"GlobalRad Ave (W/m2)": f"GHI_rt_{roofname}"})
        if i == 0:
            df_roofs = df_roof
        else:
            df_roofs = pd.merge(df_roofs, df_roof, on='Datetime', how='outer')

    for i, refname in enumerate(refnames):
        df_ref = pd.read_csv(f"{refdir}/{refname}_{monthtag}.csv")
        df_ref["Datetime"] = pd.to_datetime(df_ref["Date | Time"])
        df_ref = df_ref[["Datetime", "Global Irradiation 1 min (w/m2)", "Diffuse Irradiation 1 min (w/m2)"]]
        df_ref = df_ref.rename(columns={"Global Irradiation 1 min (w/m2)":f"GHI_ref_{refname}", "Diffuse Irradiation 1 min (w/m2)":f"SDI_ref_{refname}"})
        if i == 0:
            df_refs = df_ref
        else:
            df_refs = pd.merge(df_refs, df_ref, on='Datetime', how='outer')

    df_tars.set_index('Datetime', inplace=True)
    df_tars = df_tars.tz_localize(tz=time_zone)
    df_roofs.set_index('Datetime', inplace=True)
    df_roofs = df_roofs.tz_localize(tz=time_zone)
    df_refs.set_index('Datetime', inplace=True)
    df_refs = df_refs.tz_localize(tz=time_zone)


    tus = Location(lat, lon, time_zone)
    # irradiance = tus.get_clearsky(df_refs.index)
    solar_position = solarposition.get_solarposition(df_refs.index, lat, lon)
    solar_position = solar_position[['elevation', 'zenith', 'azimuth']]
    # solar_position = solar_position.rename(columns={'elevation':f'elevation_ref_{refname}', 'zenith':f'zenith_ref_{refname}', 'azimuth':f'azimuth_ref_{refname}'})

    df_join = pd.merge(df_tars, df_roofs, on='Datetime', how='outer')
    df_join = pd.merge(df_join, df_refs, on='Datetime', how='outer')
    df_join = pd.merge(df_join, solar_position, on='Datetime', how='outer')

    # df_refs = pd.concat([df_refs, solar_position], axis=1)
    # df_join = pd.concat([df_refs, df_roofs, df_tars], axis=1)

    for i, roofname in enumerate(roofnames):
        erbs = pvlib.irradiance.erbs(df_join[f"GHI_rt_{roofname}"], df_join["zenith"], df_join.index)
        df_join[f'DNI_rt_erbs_{roofname}'] = erbs["dni"]
        df_join[f'SDI_rt_erbs_{roofname}'] = erbs["dhi"]

    for i, refname in enumerate(refnames):
        df_join[f"DNI_ref_{refname}"] = (df_join[f"GHI_ref_{refname}"] - df_join[f"SDI_ref_{refname}"]) / np.sin(np.radians(df_join["elevation"]))
        df_join.loc[df_join[f"GHI_ref_{refname}"] < df_join[f"SDI_ref_{refname}"], f"DNI_ref_{refname}"] = 0
    
    return df_join, tarnames, roofnames, location_dict

# def join_irradiance_data_months(basedir, monthtags, lat, lon):

#     df_locations = pd.read_csv(f'{basedir}/locations.csv')# Create a nested dictionary with 'location' as the key
#     location_dict = {row['location']: {'nearest_roof': row['nearest_roof'], 'azimuth_offset': row['azimuth_offset']} for index, row in df_locations.iterrows()}
#     tarnames = df_locations['location'].unique().tolist()
#     roofnames = df_locations['nearest_roof'].unique().tolist()
    
#     time_zone = pytz.timezone('Asia/Singapore')

#     datadir = f"{basedir}/irradiance/data"

#     for i, tarname in enumerate(tarnames):
#         df_tar_list = []
#         for monthtag in monthtags:
#             if os.path.exists(f"{datadir}/NUS_{tarname}_{monthtag}.csv"):
#                 df_tar_m = pd.read_csv(f"{datadir}/NUS_{tarname}_{monthtag}.csv")
#                 df_tar_m["Datetime"] = pd.to_datetime(df_tar_m["Datetime"])
#                 df_tar_m = df_tar_m[["Datetime", "GlobalRad Ave (W/m2)"]]
#                 df_tar_m = df_tar_m.rename(columns={"GlobalRad Ave (W/m2)": f"GHI_utc_{tarname}"})
#                 df_tar_list.append(df_tar_m)
#             else:
#                 continue
#         if len(df_tar_list)>1:
#             df_tar = pd.concat(df_tar_list, axis=0, ignore_index=True)
#         else:
#             df_tar = df_tar_list[0]

#         if i == 0:
#             df_tars = df_tar
#         else:
#             df_tars = pd.merge(df_tars, df_tar, on='Datetime', how='outer')

#     for i, roofname in enumerate(roofnames):
#         df_roof_list = []
#         for monthtag in monthtags:
#             if os.path.exists(f"{datadir}/NUS_{roofname}_{monthtag}.csv"):
#                 df_roof_m = pd.read_csv(f"{datadir}/NUS_{roofname}_{monthtag}.csv")
#                 df_roof_m["Datetime"] = pd.to_datetime(df_roof_m["Datetime"])
#                 df_roof_m = df_roof_m[["Datetime", "GlobalRad Ave (W/m2)"]]
#                 df_roof_m = df_roof_m.rename(columns={"GlobalRad Ave (W/m2)": f"GHI_rt_{roofname}"})
#                 df_roof_list.append(df_roof_m)
#             else:
#                 continue
#         if len(df_roof_list)>1:
#             df_roof = pd.concat(df_roof_list, axis=0, ignore_index=True)
#         else:
#             df_roof = df_roof_list[0]
#         if i == 0:
#             df_roofs = df_roof
#         else:
#             df_roofs = pd.merge(df_roofs, df_roof, on='Datetime', how='outer')

#     # for i, refname in enumerate(refnames):
#     #     df_ref = pd.read_csv(f"{refdir}/{refname}_{monthtag}.csv")
#     #     df_ref["Datetime"] = pd.to_datetime(df_ref["Date | Time"])
#     #     df_ref = df_ref[["Datetime", "Global Irradiation 1 min (w/m2)", "Diffuse Irradiation 1 min (w/m2)"]]
#     #     df_ref = df_ref.rename(columns={"Global Irradiation 1 min (w/m2)":f"GHI_ref_{refname}", "Diffuse Irradiation 1 min (w/m2)":f"SDI_ref_{refname}"})
#     #     if i == 0:
#     #         df_refs = df_ref
#     #     else:
#     #         df_refs = pd.merge(df_refs, df_ref, on='Datetime', how='outer')

#     df_tars.set_index('Datetime', inplace=True)
#     df_tars = df_tars.tz_localize(tz=time_zone)
#     df_roofs.set_index('Datetime', inplace=True)
#     df_roofs = df_roofs.tz_localize(tz=time_zone)


#     # solar_position = solar_position.rename(columns={'elevation':f'elevation_ref_{refname}', 'zenith':f'zenith_ref_{refname}', 'azimuth':f'azimuth_ref_{refname}'})
#     df_join = pd.merge(df_tars, df_roofs, on='Datetime', how='outer')

#     # tus = Location(lat, lon, time_zone)
#     # irradiance = tus.get_clearsky(df_refs.index)
#     solar_position = solarposition.get_solarposition(df_join.index, lat, lon)
#     solar_position = solar_position[['elevation', 'zenith', 'azimuth']]

#     df_join = pd.merge(df_join, solar_position, on='Datetime', how='outer')

#     # df_refs = pd.concat([df_refs, solar_position], axis=1)
#     # df_join = pd.concat([df_refs, df_roofs, df_tars], axis=1)

#     for i, roofname in enumerate(roofnames):
#         erbs = pvlib.irradiance.erbs(df_join[f"GHI_rt_{roofname}"], df_join["zenith"], df_join.index)
#         df_join[f'DNI_rt_erbs_{roofname}'] = erbs["dni"]
#         df_join[f'SDI_rt_erbs_{roofname}'] = erbs["dhi"]

#     # for i, refname in enumerate(refnames):
#     #     df_join[f"DNI_ref_{refname}"] = (df_join[f"GHI_ref_{refname}"] - df_join[f"SDI_ref_{refname}"]) / np.sin(np.radians(df_join["elevation"]))
#     #     df_join.loc[df_join[f"GHI_ref_{refname}"] < df_join[f"SDI_ref_{refname}"], f"DNI_ref_{refname}"] = 0
    
#     return df_join, tarnames, roofnames, location_dict

def join_irradiance_data_months(basedir, monthtags, lat, lon, radmodel='erbs'):

    df_locations = pd.read_csv(f'{basedir}/locations.csv')# Create a nested dictionary with 'location' as the key
    location_dict = {row['location']: {'nearest_roof': row['nearest_roof'], 'azimuth_offset': row['azimuth_offset']} for index, row in df_locations.iterrows()}
    tarnames = df_locations['location'].unique().tolist()
    roofnames = df_locations['nearest_roof'].unique().tolist()
    
    time_zone = pytz.timezone('Asia/Singapore')

    datadir = f"{basedir}/irradiance/data"

    for i, tarname in enumerate(tarnames):
        df_tar_list = []
        for monthtag in monthtags:
            if "MWS" in tarname:
                if os.path.exists(f"{datadir}/{tarname}_{monthtag}.csv"):
                    df_tar_m = pd.read_csv(f"{datadir}/{tarname}_{monthtag}.csv")
                    df_tar_m["Datetime"] = pd.to_datetime(df_tar_m["Date Time"])
                    df_tar_m = df_tar_m[["Datetime", "Solar radiation (W/m²)"]]
                    df_tar_m = df_tar_m.rename(columns={"Solar radiation (W/m²)": f"GHI_utc_{tarname}"})
                    df_tar_list.append(df_tar_m)
                else:
                    continue
            else:
                if os.path.exists(f"{datadir}/NUS_{tarname}_{monthtag}.csv"):
                    df_tar_m = pd.read_csv(f"{datadir}/NUS_{tarname}_{monthtag}.csv")
                    df_tar_m["Datetime"] = pd.to_datetime(df_tar_m["Datetime"])
                    df_tar_m = df_tar_m[["Datetime", "GlobalRad Ave (W/m2)"]]
                    df_tar_m = df_tar_m.rename(columns={"GlobalRad Ave (W/m2)": f"GHI_utc_{tarname}"})
                    df_tar_list.append(df_tar_m)
                else:
                    continue
        if len(df_tar_list)>1:
            df_tar = pd.concat(df_tar_list, axis=0, ignore_index=True)
        else:
            df_tar = df_tar_list[0]

        if i == 0:
            df_tars = df_tar
        else:
            df_tars = pd.merge(df_tars, df_tar, on='Datetime', how='outer')

    for i, roofname in enumerate(roofnames):
        df_roof_list = []
        for monthtag in monthtags:
            if os.path.exists(f"{datadir}/NUS_{roofname}_{monthtag}.csv"):
                df_roof_m = pd.read_csv(f"{datadir}/NUS_{roofname}_{monthtag}.csv")
                df_roof_m["Datetime"] = pd.to_datetime(df_roof_m["Datetime"])
                df_roof_m = df_roof_m[["Datetime", "GlobalRad Ave (W/m2)"]]
                df_roof_m = df_roof_m.rename(columns={"GlobalRad Ave (W/m2)": f"GHI_rt_{roofname}"})
                df_roof_list.append(df_roof_m)
            else:
                continue
        if len(df_roof_list)>1:
            df_roof = pd.concat(df_roof_list, axis=0, ignore_index=True)
        else:
            df_roof = df_roof_list[0]
        if i == 0:
            df_roofs = df_roof
        else:
            df_roofs = pd.merge(df_roofs, df_roof, on='Datetime', how='outer')

    # for i, refname in enumerate(refnames):
    #     df_ref = pd.read_csv(f"{refdir}/{refname}_{monthtag}.csv")
    #     df_ref["Datetime"] = pd.to_datetime(df_ref["Date | Time"])
    #     df_ref = df_ref[["Datetime", "Global Irradiation 1 min (w/m2)", "Diffuse Irradiation 1 min (w/m2)"]]
    #     df_ref = df_ref.rename(columns={"Global Irradiation 1 min (w/m2)":f"GHI_ref_{refname}", "Diffuse Irradiation 1 min (w/m2)":f"SDI_ref_{refname}"})
    #     if i == 0:
    #         df_refs = df_ref
    #     else:
    #         df_refs = pd.merge(df_refs, df_ref, on='Datetime', how='outer')

    df_tars.set_index('Datetime', inplace=True)
    df_tars = df_tars.tz_localize(tz=time_zone)
    df_roofs.set_index('Datetime', inplace=True)
    df_roofs = df_roofs.tz_localize(tz=time_zone)


    # solar_position = solar_position.rename(columns={'elevation':f'elevation_ref_{refname}', 'zenith':f'zenith_ref_{refname}', 'azimuth':f'azimuth_ref_{refname}'})
    df_join = pd.merge(df_tars, df_roofs, on='Datetime', how='outer')

    # tus = Location(lat, lon, time_zone)
    # irradiance = tus.get_clearsky(df_refs.index)
    solar_position = solarposition.get_solarposition(df_join.index, lat, lon)
    solar_position = solar_position[['elevation', 'zenith', 'azimuth']]

    df_join = pd.merge(df_join, solar_position, on='Datetime', how='outer')

    # df_refs = pd.concat([df_refs, solar_position], axis=1)
    # df_join = pd.concat([df_refs, df_roofs, df_tars], axis=1)

    for i, roofname in enumerate(roofnames):
        if radmodel == 'erbs':
            erbs = pvlib.irradiance.erbs(df_join[f"GHI_rt_{roofname}"], df_join["zenith"], df_join.index)
        elif radmodel == 'erbs_driesse':
            erbs = pvlib.irradiance.erbs_driesse(df_join[f"GHI_rt_{roofname}"], df_join["zenith"], df_join.index)
        df_join[f'DNI_rt_{radmodel}_{roofname}'] = erbs["dni"]
        df_join[f'SDI_rt_{radmodel}_{roofname}'] = erbs["dhi"]

    # for i, refname in enumerate(refnames):
    #     df_join[f"DNI_ref_{refname}"] = (df_join[f"GHI_ref_{refname}"] - df_join[f"SDI_ref_{refname}"]) / np.sin(np.radians(df_join["elevation"]))
    #     df_join.loc[df_join[f"GHI_ref_{refname}"] < df_join[f"SDI_ref_{refname}"], f"DNI_ref_{refname}"] = 0
    
    return df_join, tarnames, roofnames, location_dict

def filter_stable_condition(df, consective = 10, threshold = 100):
    # Filter the columns that match "GHI_ref_*" and "GHI_roof_**" patterns
    # ghi_columns = [col for col in df.columns if 'GHI_ref' in col or 'GHI_rt' in col]
    ghi_columns = [col for col in df.columns if 'GHI_rt' in col]

    # Calculate the min, max, and difference for the selected columns
    df['min_value'] = df[ghi_columns].min(axis=1)
    df['max_value'] = df[ghi_columns].max(axis=1)
    df['diff'] = df['max_value'] - df['min_value']
    # df['diff_pct_of_max'] = df['diff'] / df['max_value']

    # Define the threshold for difference (10% of max value)
    # threshold = 0.10
    # threshold = 100

    # Find rows where the difference is less than 10% of the maximum value
    # df['condition_met'] = df['diff_pct_of_max'] < threshold
    df['condition_met'] = df['diff'] < threshold

    # Find sequences of more than 10 consecutive rows meeting the condition
    df['group'] = (df['condition_met'] != df['condition_met'].shift()).cumsum()
    df_filtered = df[df['condition_met']].groupby('group').filter(lambda x: len(x) > consective)

    # df_filtered = df_filtered[df_filtered["GHI_ref_S24_changi"]>100]

    # Resulting DataFrame with sequences meeting the criteria
    return df_filtered