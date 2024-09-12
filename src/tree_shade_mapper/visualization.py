
from PIL import Image, ImageDraw
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import contextily as ctx
import seaborn as sns
import pandas as pd
import geopandas as gpd
import os
import shutil
import math

import folium
import branca.colormap as cm
import io
import base64
from PIL import Image

from geojson import Feature, FeatureCollection
import json
import matplotlib
import h3

from .image_process import get_transmittance_center_of_modes, get_transmittance_center_of_modes_upper#, get_transmittance_seg, get_transmittance_bin
from .image_process import get_sky_view_factor_from_binary


def display_sun_trajectory(frame_path, df_sunposi, azimuth_offset=0, mode = "point"):
    img = Image.open(frame_path)
    img = img.resize((2048, 1024))

    img_arr = np.array(img)

    img_h = img_arr.shape[0]
    img_w = img_arr.shape[1]

    # Draw markers
    x = [10]
    y = [20]
    draw = ImageDraw.Draw(img)
    #font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf', 30)

    time_list = df_sunposi.index.strftime('%Y-%m-%d %H:%M').tolist()

    for i in range(len(df_sunposi)):
        azimuth = (df_sunposi["azimuth"][i]+azimuth_offset)%360#degree
        zenith = df_sunposi["zenith"][i]#degree

        #print(azimuth, zenith)
        if zenith < 90:
            if mode == "trajectory":
                if (time_list[i][-2:] == '00')|(time_list[i][-2:] == '30'):
                    index_azimuth = int(round(img_w * azimuth / 360, 0))
                    index_zenith = int(round(img_h / 2 * zenith / 90, 0))
                    #print(index_azimuth, index_azimuth%360)
                    radius=5
                    draw.ellipse((index_azimuth-radius, index_zenith-radius, index_azimuth+radius, index_zenith+radius), outline='yellow', fill='yellow')
                    draw.text((index_azimuth+radius*2, index_zenith), time_list[i], fill='red')#, font=font)
            if mode == "point":
                index_azimuth = int(round(img_w * azimuth / 360, 0))
                index_zenith = int(round(img_h / 2 * zenith / 90, 0))
                #print(index_azimuth, index_azimuth%360)
                radius=5
                draw.ellipse((index_azimuth-radius, index_zenith-radius, index_azimuth+radius, index_zenith+radius), outline='yellow', fill='yellow')
                draw.text((index_azimuth+radius*2, index_zenith), time_list[i], fill='red')#, font=font)

    display(img)

def display_sun_trajectory_with_value(frame_path, df_sunposi, azimuth_offset=0, value = "error"):
    img = Image.open(frame_path)
    img = img.resize((2048, 1024))

    img_arr = np.array(img)

    img_h = img_arr.shape[0]
    img_w = img_arr.shape[1]

    # Draw markers
    x = [10]
    y = [20]
    draw = ImageDraw.Draw(img)
    #font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf', 30)

    time_list = df_sunposi.index.strftime('%Y-%m-%d %H:%M').tolist()
    # norm = Normalize(vmin=df['value'].min(), vmax=df['value'].max())
    for i in range(len(df_sunposi)):
        azimuth = (df_sunposi["azimuth"][i]+azimuth_offset)%360#degree
        zenith = df_sunposi["zenith"][i]#degree
        color = value_to_color(df_sunposi[value][i])

        #print(azimuth, zenith)
        if zenith < 90:
            index_azimuth = int(round(img_w * azimuth / 360, 0))
            index_zenith = int(round(img_h / 2 * zenith / 90, 0))
            #print(index_azimuth, index_azimuth%360)
            radius=5
            draw.ellipse((index_azimuth-radius, index_zenith-radius, index_azimuth+radius, index_zenith+radius), outline=color, fill=color)
            # draw.text((index_azimuth+radius*2, index_zenith), time_list[i], fill='red')#, font=font)

    display(img)

def value_to_color(value):
    norm = matplotlib.colors.Normalize(vmin=-500, vmax=500)
    # return tuple(int(255 * x) for x in matplotlib.cm.viridis(norm(value))[:3])
    return tuple(int(255 * x) for x in matplotlib.cm.bwr(norm(value))[:3])

def visualize_solar(df_solar, columns, prediction = None, transmittance = None):
    # Plotting the line chart
    plt.figure(figsize=(8, 3))
    for column in columns:
        plt.plot(df_solar.index, df_solar[column], label=column)
    if prediction:
        plt.plot(df_solar.index, df_solar[prediction], label='Solar Radiation prediction')
    #plt.plot(df_concat.index, df_concat["direct_normal_erbs"], label='direct_normal_erbs')
    #plt.plot(df_concat.index, df_concat['sky_diffuse_erbs'], label='sky_diffuse_erbs')

    plt.title('Solar Radiation Comparison')
    plt.xlabel('Time')
    plt.ylabel('Solar Radiation')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Plotting the line chart
    if transmittance:
        plt.figure(figsize=(8, 2))
        plt.plot(df_solar.index, df_solar[transmittance], label='transmittance')
        #plt.plot(df_concat.index, df_concat["apparent_zenith"], label='zenith')

        plt.title('change of transmittance')
        plt.xlabel('Time')
        plt.ylabel('transmittance')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    if prediction:
        ax = df_solar.plot.scatter( 
            x='solar_radiation_ws',
            y='gsi_utc',
            c='DarkBlue')

def display_image(array, colarmap):
    plt.figure(figsize=(5,5))
    plt.imshow(array, cmap=colarmap)
    plt.axis('off')
    plt.show()

# Add custom basemaps to folium
basemaps = {
    'Google Maps': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Maps',
        overlay = True,
        control = True
    ),
    'Google Satellite': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Google Terrain': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Terrain',
        overlay = True,
        control = True
    ),
    'Google Satellite Hybrid': folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
    ),
    'Esri Satellite': folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = True,
        control = True
    ),
    'Light Map': folium.TileLayer(
        tiles = 'cartodbpositron',
        attr = 'cartodb',
        name = 'Light Map',
        overlay = True,
        control = True
    ),
    'Dark Map': folium.TileLayer(
        tiles = 'cartodbdark_matter',
        attr = 'cartodb',
        name = 'Dark Map',
        overlay = True,
        control = True
    )
}

def mapping_points_image_popup(df_map, value_name, vmin, vmax, imgdir):
    m = folium.Map(location=[df_map['lat'].mean(), df_map['lon'].mean()], zoom_start=18)

    folium.TileLayer('cartodbpositron',name="Light Map",control=False).add_to(m)
    folium.TileLayer('cartodbdark_matter',name="Dark Map",control=False).add_to(m)

    # Add custom basemaps
    basemaps['Google Maps'].add_to(m)
    basemaps['Light Map'].add_to(m)
    #basemaps['Dark Map'].add_to(m)
    #basemaps['Google Satellite Hybrid'].add_to(m)

    # create color range based on values
    #colormap = folium.LinearColormap(colors=['red', 'yellow', 'green'], vmin=df_map[item1].min(), vmax=df_map[item1].max())
    #colormap = folium.LinearColormap(colors=['red', 'yellow', 'green'], vmin=0, vmax=df_map[item1].max())
    #colormap = folium.LinearColormap(colors=['blue', 'white', 'red'], vmin=31, vmax=35)
    #colormap = cm.linear.plasma.scale(31, 35)
    colormap = cm.linear.viridis.scale(vmin, vmax)
    # colormap = cm.LinearColormap(colors=['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red'],
    #                              index=[30, 31, 32, 33, 34, 35], vmin=30, vmax=35,
    #                              caption='Total Standard deviation at the point[mm]')

    # add markers to map with color based on value range
    for i in range(len(df_map)):
        #for i in range(10):
        frame_path = os.path.join(imgdir, df_map["panoIMGname_target"][i])
        img = Image.open(frame_path)
        new_img = img.resize((256, 128))  # x, y
        buffer = io.BytesIO()
        new_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue())
        html = '<img src="data:image/png;base64,{}">'.format
        iframe = folium.IFrame(html(encoded.decode('UTF-8')), width=280, height=140)
        popup = folium.Popup(iframe, max_width=280)
        #html = '''frame_id:''' + str(df_map["frame_key"][i])
        #iframe = folium.IFrame(html, width=300, height=100)
        #popup = folium.Popup(iframe, max_width=300)
        folium.CircleMarker(location=[df_map.iloc[i]['lat'], df_map.iloc[i]['lon']], radius=0.1,
                                color=colormap(df_map.iloc[i][value_name]), fill=True, fill_opacity=1, popup=popup).add_to(m)

    m.add_child(colormap)

    #for coord in latlong:
    #  html = '''緯度:''' + str(coord[0]) + '''<br>''' + '''経度:''' + str(coord[1])
    #
    #  iframe = folium.IFrame(html, width=300, height=100)
    #  popup = folium.Popup(iframe, max_width=300)
    #  #folium.Marker( location=[ coord[0], coord[1] ], fill_color='#43d9de', radius=1, popup=popup).add_to( mapit )
    #  folium.Circle(name='Emosca', location=[ coord[0], coord[1] ], fill_color='#43d9de', radius=1, popup=popup).add_to( m )

    folium.LayerControl().add_to(m)

    return m

def mapping_points_with_value(df_map, value_name, vmin, vmax, cmap="viridis", save_as=None):

    m = folium.Map(location=[df_map['lat'].mean(), df_map['lon'].mean()], zoom_start=18)

    folium.TileLayer('cartodbpositron',name="Light Map",control=False).add_to(m)
    folium.TileLayer('cartodbdark_matter',name="Dark Map",control=False).add_to(m)

    # Add custom basemaps
    basemaps['Google Maps'].add_to(m)
    basemaps['Light Map'].add_to(m)
    #basemaps['Dark Map'].add_to(m)
    #basemaps['Google Satellite Hybrid'].add_to(m)

    # create color range based on values
    #colormap = folium.LinearColormap(colors=['red', 'yellow', 'green'], vmin=df_map[item1].min(), vmax=df_map[item1].max())
    #colormap = folium.LinearColormap(colors=['red', 'yellow', 'green'], vmin=0, vmax=df_map[item1].max())
    #colormap = folium.LinearColormap(colors=['blue', 'white', 'red'], vmin=31, vmax=35)
    #colormap = cm.linear.plasma.scale(31, 35)
    if cmap == "viridis":
        colormap = cm.linear.viridis.scale(vmin, vmax)
    if cmap == "inferno":
        colormap = cm.linear.inferno.scale(vmin, vmax)
    if cmap == "magma":
        colormap = cm.linear.magma.scale(vmin, vmax)
    if cmap == "plasma":
        colormap = cm.linear.plasma.scale(vmin, vmax)
    # colormap = cm.LinearColormap(colors=['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red'],
    #                              index=[30, 31, 32, 33, 34, 35], vmin=30, vmax=35,
    #                              caption='Total Standard deviation at the point[mm]')

    # add markers to map with color based on value range
    for i in range(len(df_map)):
        #for i in range(10):
        folium.CircleMarker(location=[df_map.iloc[i]['lat'], df_map.iloc[i]['lon']], radius=0.05,
                                color=colormap(df_map.iloc[i][value_name]), fill=True, fill_opacity=1).add_to(m)

    m.add_child(colormap)

    #for coord in latlong:
    #  html = '''緯度:''' + str(coord[0]) + '''<br>''' + '''経度:''' + str(coord[1])
    #
    #  iframe = folium.IFrame(html, width=300, height=100)
    #  popup = folium.Popup(iframe, max_width=300)
    #  #folium.Marker( location=[ coord[0], coord[1] ], fill_color='#43d9de', radius=1, popup=popup).add_to( mapit )
    #  folium.Circle(name='Emosca', location=[ coord[0], coord[1] ], fill_color='#43d9de', radius=1, popup=popup).add_to( m )

    folium.LayerControl().add_to(m)

    if save_as:
        m.save(save_as)

    return m

def mapping_points_image_popup(df_map, value_name, vmin, vmax, imgdir):
    m = folium.Map(location=[df_map['lat'].mean(), df_map['lon'].mean()], zoom_start=18)

    folium.TileLayer('cartodbpositron',name="Light Map",control=False).add_to(m)
    folium.TileLayer('cartodbdark_matter',name="Dark Map",control=False).add_to(m)

    # Add custom basemaps
    basemaps['Google Maps'].add_to(m)
    basemaps['Light Map'].add_to(m)
    #basemaps['Dark Map'].add_to(m)
    #basemaps['Google Satellite Hybrid'].add_to(m)

    # create color range based on values
    #colormap = folium.LinearColormap(colors=['red', 'yellow', 'green'], vmin=df_map[item1].min(), vmax=df_map[item1].max())
    #colormap = folium.LinearColormap(colors=['red', 'yellow', 'green'], vmin=0, vmax=df_map[item1].max())
    #colormap = folium.LinearColormap(colors=['blue', 'white', 'red'], vmin=31, vmax=35)
    #colormap = cm.linear.plasma.scale(31, 35)
    colormap = cm.linear.viridis.scale(vmin, vmax)
    # colormap = cm.LinearColormap(colors=['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red'],
    #                              index=[30, 31, 32, 33, 34, 35], vmin=30, vmax=35,
    #                              caption='Total Standard deviation at the point[mm]')

    # add markers to map with color based on value range
    for i in range(len(df_map)):
        #for i in range(10):
        frame_path = os.path.join(imgdir, f'{df_map["frame_key"][i]}.jpg')
        img = Image.open(frame_path)
        xyratio = img.size[1] / img.size[0]
        newx = 256
        new_img = img.resize((newx, int(newx * xyratio)))  # x, y
        buffer = io.BytesIO()
        new_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue())
        html = '<img src="data:image/png;base64,{}">'.format
        iframe = folium.IFrame(html(encoded.decode('UTF-8')), width=int(newx*1.1), height=int(newx*xyratio*1.1))
        popup = folium.Popup(iframe, max_width=280)
        #html = '''frame_id:''' + str(df_map["frame_key"][i])
        #iframe = folium.IFrame(html, width=300, height=100)
        #popup = folium.Popup(iframe, max_width=300)
        folium.CircleMarker(location=[df_map.iloc[i]['lat'], df_map.iloc[i]['lon']], radius=0.1,
                                color=colormap(df_map.iloc[i][value_name]), fill=True, fill_opacity=1, popup=popup).add_to(m)

    m.add_child(colormap)

    #for coord in latlong:
    #  html = '''緯度:''' + str(coord[0]) + '''<br>''' + '''経度:''' + str(coord[1])
    #
    #  iframe = folium.IFrame(html, width=300, height=100)
    #  popup = folium.Popup(iframe, max_width=300)
    #  #folium.Marker( location=[ coord[0], coord[1] ], fill_color='#43d9de', radius=1, popup=popup).add_to( mapit )
    #  folium.Circle(name='Emosca', location=[ coord[0], coord[1] ], fill_color='#43d9de', radius=1, popup=popup).add_to( m )

    folium.LayerControl().add_to(m)

    return m

def mapping_points(df_map):

    m = folium.Map(location=[df_map['lat'].mean(), df_map['lon'].mean()], zoom_start=18)

    folium.TileLayer('cartodbpositron',name="Light Map",control=False).add_to(m)
    folium.TileLayer('cartodbdark_matter',name="Dark Map",control=False).add_to(m)

    # Add custom basemaps
    basemaps['Google Maps'].add_to(m)
    basemaps['Light Map'].add_to(m)
    #basemaps['Dark Map'].add_to(m)
    #basemaps['Google Satellite Hybrid'].add_to(m)

    # add markers to map with color based on value range
    for i in range(len(df_map)):
        #for i in range(10):
        folium.CircleMarker(location=[df_map.iloc[i]['lat'], df_map.iloc[i]['lon']], radius=0.1,
                                color="b", fill=True, fill_opacity=1).add_to(m)

    folium.LayerControl().add_to(m)

    return m

def hexagons_dataframe_to_geojson(df_hex, file_output = None, column_name = "value"):
    """
    Produce the GeoJSON for a dataframe, constructing the geometry from the "hex_id" column
    and with a property matching the one in column_name
    """    
    list_features = []
    
    for i, row in df_hex.iterrows():
        try:
            geometry_for_row = { "type" : "Polygon", "coordinates": [h3.h3_to_geo_boundary(h=row["hex_id"],geo_json=True)]}
            feature = Feature(geometry = geometry_for_row , id=row["hex_id"], properties = {column_name : row[column_name]})
            list_features.append(feature)
        except:
            print("An exception occurred for hex " + row["hex_id"]) 

    feat_collection = FeatureCollection(list_features)
    geojson_result = json.dumps(feat_collection)
    return geojson_result

def get_color(custom_cm, val, vmin, vmax):
    return matplotlib.colors.to_hex(custom_cm((val-vmin)/(vmax-vmin)))

def choropleth_map(df_aggreg, df_original, value_name, vmin, vmax, fill_opacity = 0.9, cmap = None, initial_map = None):
    """
    Creates choropleth maps given the aggregated data. initial_map can be an existing map to draw on top of.
    """    
    column_name = value_name

    #colormap
    min_value = df_aggreg[column_name].min()
    max_value = df_aggreg[column_name].max()
    mean_value = df_aggreg[column_name].mean()
    print(f"Colour column min value {min_value}, max value {max_value}, mean value {mean_value}")
    print(f"Hexagon cell count: {df_aggreg['hex_id'].nunique()}")
    
    # the name of the layer just needs to be unique, put something silly there for now:
    name_layer = "Choropleth " + str(df_aggreg)
    
    if initial_map is None:
        initial_map = folium.Map(location= [df_original['lat'].mean(), df_original['lon'].mean()], zoom_start=16, tiles="cartodbpositron")

    #create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex = df_aggreg, column_name = column_name)

    # color_map_name 'Blues' for now, many more at https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose from!
    #colormap = matplotlib.cm.get_cmap(cmap)
    colormap = cm.LinearColormap(colors=plt.cm.plasma.colors, vmin=vmin, vmax=vmax)
    if cmap == "plasma":
        colormap = cm.LinearColormap(colors=plt.cm.plasma.colors, vmin=vmin, vmax=vmax)
    elif cmap == "inferno":
        colormap = cm.LinearColormap(colors=plt.cm.inferno.colors, vmin=vmin, vmax=vmax)
    elif cmap == "magma":
        colormap = cm.LinearColormap(colors=plt.cm.magma.colors, vmin=vmin, vmax=vmax)
    elif cmap == "Blues":
        num_samples = 256  # Number of samples to take from the colormap
        sample_points = [i/num_samples for i in range(num_samples)]
        colors = [plt.cm.Blues(point) for point in sample_points]
        # Creating the colormap
        colormap = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax)
    # elif cmap == "binary":
    #     colormap = cm.LinearColormap(colors=plt.cm.binary.colors, vmin=vmin, vmax=vmax)
    else:
        colormap = cm.LinearColormap(colors=plt.cm.viridis.colors, vmin=vmin, vmax=vmax)

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            #'fillColor': get_color(colormap, feature['properties'][column_name], vmin=vmin, vmax=vmax),
            'fillColor': colormap(feature['properties'][column_name]),
            #'color': border_color,
            'weight': 0,
            'fillOpacity': fill_opacity 
        }, 
        name = name_layer
    ).add_to(initial_map)

    # Add color scale as a caption to the map
    colormap.caption = value_name
    colormap.add_to(initial_map)

    return initial_map

def save_choropleth_svg(df_aggreg, df_original, value_name, vmin, vmax, fill_opacity = 0.7, cmap = None, initial_map = None, save_dir = None, save_name = None):
    """
    Creates choropleth maps given the aggregated data. initial_map can be an existing map to draw on top of.
    """    
    column_name = value_name

    #colormap
    min_value = df_aggreg[column_name].min()
    max_value = df_aggreg[column_name].max()
    mean_value = df_aggreg[column_name].mean()
    print(f"Colour column min value {min_value}, max value {max_value}, mean value {mean_value}")
    print(f"Hexagon cell count: {df_aggreg['hex_id'].nunique()}")
    
    # the name of the layer just needs to be unique, put something silly there for now:
    name_layer = "Choropleth " + str(df_aggreg)
    
    if initial_map is None:
        initial_map = folium.Map(location= [df_original['lat'].mean(), df_original['lon'].mean()], zoom_start=16, tiles="cartodbpositron")

    #create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex = df_aggreg, column_name = column_name)
    geojson_data = json.loads(geojson_data)
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

    # Plotting the data with a color map based on the 'solar_radiation_target_pred' column
    #fig, ax = plt.subplots(facecolor='none')
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.patch.set_alpha(0.5)
    gdf.plot(column=column_name, cmap=cmap, ax=ax, legend=True, vmin=vmin, vmax=vmax)

    # Save the plot as an SVG file
    if save_name:
        user_colored_svg_file = f"{save_dir}/{value_name}_{save_name}_map.svg"
    else:
        user_colored_svg_file = f"{save_dir}/{value_name}_map.svg"

    #plt.savefig(user_colored_svg_file, format='svg', transparent=True, bbox_inches='tight')
    plt.savefig(user_colored_svg_file, format='svg')

    # color_map_name 'Blues' for now, many more at https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose from!
    #colormap = matplotlib.cm.get_cmap(cmap)

    # colormap = cm.LinearColormap(colors=plt.cm.plasma.colors, vmin=vmin, vmax=vmax)
    # if cmap == "plasma":
    #     colormap = cm.LinearColormap(colors=plt.cm.plasma.colors, vmin=vmin, vmax=vmax)
    # elif cmap == "inferno":
    #     colormap = cm.LinearColormap(colors=plt.cm.inferno.colors, vmin=vmin, vmax=vmax)
    # elif cmap == "magma":
    #     colormap = cm.LinearColormap(colors=plt.cm.magma.colors, vmin=vmin, vmax=vmax)
    # # elif cmap == "binary":
    # #     colormap = cm.LinearColormap(colors=plt.cm.binary.colors, vmin=vmin, vmax=vmax)
    # else:
    #     colormap = cm.LinearColormap(colors=plt.cm.viridis.colors, vmin=vmin, vmax=vmax)

def save_choropleth_with_basemap(df_aggreg, df_original, value_name, vmin, vmax, fill_opacity=0.7, cmap=None, initial_map=None, save_dir=None):
    """
    Creates choropleth maps with a raster basemap, expands the view area, and adds a color bar on the map.
    `expand_area_ratio` determines how much to expand the area around the data.
    """
    expand_area_ratio_x = 0.25
    expand_area_ratio_y = 0.15
    column_name = value_name
    # Create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex=df_aggreg, column_name=column_name)
    geojson_data = json.loads(geojson_data)
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
    # Set coordinate reference system to Web Mercator
    gdf = gdf.set_crs("EPSG:4326")  # Assuming your data is in WGS84
    gdf = gdf.to_crs(epsg=3857)  # Convert to Web Mercator
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column=column_name, cmap=cmap, ax=ax, alpha=fill_opacity, legend=False, vmin=vmin, vmax=vmax)
    
    # Expand the area of the map
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = (maxx - minx) * expand_area_ratio_x
    dy = (maxy - miny) * expand_area_ratio_y
    ax.set_xlim(minx - dx, maxx + 2*dx)
    ax.set_ylim(miny - dy, maxy + dy)
    
    # Remove axis lines
    ax.set_axis_off()
    
    # Adding Contextily basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Add color bar on the map
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Create an inset axes for the colorbar
    cax = inset_axes(ax, width="2%", height="30%", loc='lower right', 
                    bbox_to_anchor=(-0.15, 0.05, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0)
    
    if "acc" in value_name:
        value_title = r'$GHI_{utc\_acc}$  (MJm$^{-2}$)'
    elif "ghi" in value_name:
        value_title = r'$GHI_{utc}$  (Wm$^{-2}$)'
    elif "svf" in value_name:
        value_title = 'Sky view factor'

    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label(value_title, rotation=270, labelpad=15)
    
    # Save the plot as a PNG file
    user_colored_png_file = f"{save_dir}/{value_name}_map.png"
    plt.savefig(user_colored_png_file, format='png', bbox_inches='tight', dpi=300)
    plt.show()
    # plt.close(fig)

def save_choropleth_with_basemap_time(df_aggreg, datetime, value_name, vmin, vmax, fill_opacity=0.7, cmap=None, initial_map=None, save_dir=None, save_name = None, basemap = None):
    """
    Creates choropleth maps with a raster basemap and expands the view area. 
    `expand_area_ratio` determines how much to expand the area around the data.
    """
    expand_area_ratio_x = 0.25
    expand_area_ratio_y = 0.15

    column_name = value_name

    # Create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(df_hex=df_aggreg, column_name=column_name)
    geojson_data = json.loads(geojson_data)
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

    # Set coordinate reference system to Web Mercator
    gdf = gdf.set_crs("EPSG:4326")  # Assuming your data is in WGS84
    gdf = gdf.to_crs(epsg=3857)  # Convert to Web Mercator

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(column=column_name, cmap=cmap, ax=ax, alpha=fill_opacity, legend=False, vmin=vmin, vmax=vmax)

    # Expand the area of the map
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = (maxx - minx) * expand_area_ratio_x
    dy = (maxy - miny) * expand_area_ratio_y
    ax.set_xlim(minx - dx, maxx + 2*dx)
    ax.set_ylim(miny - dy, maxy + dy)

    # Remove axis lines
    ax.set_axis_off()

    # Adding Contextily basemap
    if basemap == "carto":
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    if basemap == "satellite":
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)    

    # Add color bar on the map
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Create an inset axes for the colorbar
    cax = inset_axes(ax, width="2%", height="30%", loc='lower right', 
                    bbox_to_anchor=(-0.15, 0.05, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0)
    
    if "acc" in value_name:
        value_title = r'$GHI_{utc\_acc}$  (kJm$^{-2}$)'
    elif "ghi" in value_name:
        value_title = r'$GHI_{utc}$  (Wm$^{-2}$)'
    elif "svf" in value_name:
        value_title = 'Sky view factor'
    
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label(value_title, rotation=270, labelpad=15)

    text_content = f"{datetime}".replace(":00+08:00", "")
    text_position = (0.05, 0.05)  # Top-left corner
    text_style = {'fontsize': 15, 'color': 'black', 'fontweight': 'bold', 'ha': 'left', 'va': 'bottom'}  # Optional: override default style
    ax.text(text_position[0], text_position[1], text_content, transform=ax.transAxes, **text_style)

    # Save the plot as a PNG file
    user_colored_png_file = f"{save_dir}/{value_name}_map.png"

    # Save the plot as an SVG file
    if save_name:
        user_colored_png_file = f"{save_dir}/{value_name}_{save_name}_map.png"
    else:
        user_colored_png_file = f"{save_dir}/{value_name}_map.png"

    plt.savefig(user_colored_png_file, format='png', bbox_inches='tight')

    plt.show()
    # plt.close(fig)

def mapping_h3_grid(df_map, value_name, vmin, vmax, resolution = None, cmap = None, save_dir = None, fill_opacity = None):
    hex_ids = df_map.apply(lambda row: h3.geo_to_h3(row.lat, row.lon, resolution), axis = 1)
    df_map = df_map.assign(hex_id=hex_ids.values)
    df_h3 = df_map.groupby("hex_id", as_index=False).agg({value_name: "mean"})
    # save_choropleth_svg(df_h3, df_map, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir)
    save_choropleth_with_basemap(df_h3, df_map, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir)
    return choropleth_map(df_h3, df_map, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None)

def mapping_h3_grid_timeseries(df_map, value_name, vmin, vmax, resolution = None, cmap = None, save_dir = None, fill_opacity = None, basemap = None):
    df_map['time'] = pd.to_datetime(df_map['time'])
    times = list(set(df_map["time"].tolist()))
    times = sorted(times)

    for time in times:        
        # Convert to datetime object
        # date_obj = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        formatted_date_str = time.strftime('%Y-%m-%d-%H-%M')

        df_map_filtered = df_map[df_map['time'] == time]
        hex_ids = df_map_filtered.apply(lambda row: h3.geo_to_h3(row.lat, row.lon, resolution), axis = 1)
        df_map_filtered = df_map_filtered.assign(hex_id=hex_ids.values)
        df_h3 = df_map_filtered.groupby("hex_id", as_index=False).agg({value_name: "mean"})

        # save_choropleth_svg(df_h3, df_map_filtered, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir, save_name = formatted_date_str)
        save_choropleth_with_basemap_time(df_h3, time, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir, save_name = formatted_date_str, basemap = basemap)

def mapping_h3_grid_timeseries_normalized(df_map, value_name, resolution = None, cmap = None, save_dir = None, fill_opacity = None):
    df_map['time'] = pd.to_datetime(df_map['time'])
    times = list(set(df_map["time"].tolist()))

    for time in times:
        # Convert to datetime object
        # date_obj = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        formatted_date_str = time.strftime('%Y-%m-%d-%H')

        df_map_filtered = df_map[df_map['time'] == time]
        hex_ids = df_map_filtered.apply(lambda row: h3.geo_to_h3(row.lat, row.lon, resolution), axis = 1)
        df_map_filtered = df_map_filtered.assign(hex_id=hex_ids.values)
        df_h3 = df_map_filtered.groupby("hex_id", as_index=False).agg({value_name: "mean"})

        vmax = df_h3[value_name].max()
        vmin = df_h3[value_name].min()

        print(time, vmin, vmax)

        save_choropleth_svg(df_h3, df_map_filtered, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir, save_name = formatted_date_str)
        save_choropleth_with_basemap_time(df_h3, df_map_filtered, value_name, vmin, vmax, fill_opacity = fill_opacity, cmap = cmap, initial_map = None, save_dir = save_dir, save_name = formatted_date_str)

def process_frame(ori_path, seg_path, tra_path, bin_path, kernel_size, binary_type, model):
    array_transmittance, array_binary = get_transmittance_center_of_modes(ori_path, seg_path, kernel_size, binary_type=binary_type, model=model)
    np.save(tra_path, array_transmittance)
    np.save(bin_path, array_binary)
    sky_view_factor = get_sky_view_factor_from_binary(array_binary)
    return sky_view_factor

def rename_and_copy_files(source_dir, target_dir):
    # Create the target directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if "WS" in filename:
            if "_" in filename:
                # Split the filename at the underscore and take the first part
                new_filename = filename.split('_')[0] + os.path.splitext(filename)[1]
                # Full path for source and target files
                source_file = os.path.join(source_dir, filename)
                target_file = os.path.join(target_dir, new_filename)

                # Copy the file from source to target with the new name
                shutil.copy(source_file, target_file)
                print(f"Renamed and copied: {filename} to {new_filename}")
            else:
                # If no underscore, just copy with the same name
                shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, filename))
                print(f"Copied without renaming: {filename}")
        else:
            # If no underscore, just copy with the same name
            shutil.copy(os.path.join(source_dir, filename), os.path.join(target_dir, filename))
            print(f"Copied without renaming: {filename}")

def mapping_accu(base_dir, vmin = 0, vmax = 1000, resolution = 15, models = ["tcm"], gps_precision_lim = None):
    df_solar_path = f"{base_dir}/frames_solar_accu.csv"
    df_solar_all = pd.read_csv(df_solar_path)
    if gps_precision_lim:
        df_solar_all = df_solar_all[df_solar_all["precision"]>gps_precision_lim]

    for model in models:
        map_dir = f"{base_dir}/map_{model}_{resolution}"
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)

        item = f"ghi_utc_acc_{model}" 
        vmax = 5 * math.ceil(df_solar_all[f"ghi_utc_acc_{model}"].max() / 5)
        mapping_h3_grid(df_solar_all, item, vmin, vmax, resolution = resolution, cmap = "inferno", save_dir = map_dir, fill_opacity = 0.8)

def mapping_svf(base_dir, vmin = 0, vmax = 1, resolution = 15, models = ["tcm"], gps_precision_lim = None):
    df_svf_path = f"{base_dir}/frames_svf.csv"
    df_svf = pd.read_csv(df_svf_path)
    if gps_precision_lim:
        df_svf = df_svf[df_svf["precision"]>gps_precision_lim]

    cmap = sns.dark_palette("#4CC9F0", reverse=False, as_cmap=True)

    for model in models:
        map_dir = f"{base_dir}/map_{model}_{resolution}"
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)

        item = f"svf_{model}" 
        mapping_h3_grid(df_svf, item, vmin, vmax, resolution = resolution, cmap = cmap, save_dir = map_dir, fill_opacity = 0.8)

def mapping_time_series(base_dir, vmin = 0, vmax = 1000, resolution = 15, models = ["tcm"], gps_precision_lim = None):
    df_solar_path = f"{base_dir}/frames_solar.csv"
    df_solar_all = pd.read_csv(df_solar_path)
    if gps_precision_lim:
        df_solar_all = df_solar_all[df_solar_all["precision"]>gps_precision_lim]

    for model in models:
        map_dir = f"{base_dir}/map_{model}_{resolution}_time_series"
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)

        item = f"ghi_utc_{model}"    
        mapping_h3_grid_timeseries(df_solar_all, item, vmin, vmax, resolution = resolution, cmap = "inferno", save_dir = map_dir, fill_opacity = 0.8, basemap = "carto")

def hexbin_plot_prediction(df, ground_truth, prediction, xlabel='x', ylabel='y', xymin=None, xymax=None, vmax=None, cmap="viridis", tag = ''):
    # density_scatter(df[weather_item+'_target_pred'], df[weather_item+'_target'], bins = [10,10] )
    # density_scatter(df[weather_item+'_reference'], df[weather_item+'_target'], bins = [10,10] )

    # Generate some random data
    np.random.seed(0)
    x = df[ground_truth]
    y = df[prediction]

    xyrange = xymax-xymin
    x.loc[len(x)] = xymax + xyrange
    y.loc[len(y)] = xymax + xyrange
    x.loc[len(x)] = xymin - xyrange
    y.loc[len(y)] = xymin - xyrange

    # Create a figure and two subplots, side by side
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True, figsize=(5, 4))

    #vmax = round(len(x)/50, -3)

    # First plot
    h = ax.hexbin(x, y, gridsize=100, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_xlabel('ground truth ' + r"$\mathrm{W/m}^{2}$")
    # ax.set_ylabel(f'prediction ({tag}) ' + r"$\mathrm{W/m}^{2}$")
    ax.set_xlim([xymin, xymax])
    ax.set_ylim([xymin, xymax])
    cb = fig.colorbar(h, ax=ax)
    cb.set_label('Counts')

    # Adding the x=y line to both subplots
    ax.plot([0, 1200], [0, 1200], color='white', linestyle='--')  # This plots the x=y line
    
    svg_file_path = os.path.join("figure", f"prediction_hexbin_{tag}.svg")
    plt.savefig(svg_file_path)

# def save_point_map_with_basemap(df_points, value_name, vmin, vmax, cmap='viridis', initial_map=None, save_dir=None):
#     """
#     Creates a map with points plotted on it, colored according to their values. 
#     """
#     # Convert the DataFrame to a GeoDataFrame
#     gdf = gpd.GeoDataFrame(
#         df_points, 
#         geometry=gpd.points_from_xy(df_points.lon, df_points.lat),
#         crs='EPSG:4326'
#     )
    
#     # Convert to Web Mercator projection
#     gdf = gdf.to_crs(epsg=3857)
    
#     # Plotting
#     fig, ax = plt.subplots(figsize=(10, 10))
#     scatter = gdf.plot(
#         ax=ax, 
#         column=value_name, 
#         cmap=cmap, 
#         alpha=0.7, 
#         legend=False, 
#         vmin=vmin, 
#         vmax=vmax,
#         markersize=15,  # Adjust the size of the points
#         marker='o'      # Adjust the shape of the points
#     )

#     # Adding Contextily basemap
#     ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
#     # Remove axis lines
#     ax.set_axis_off()

#     # Expand the area of the map slightly
#     minx, miny, maxx, maxy = gdf.total_bounds
#     dx = (maxx - minx) * 0.3  # reduced the expand area ratio
#     dy = (maxy - miny) * 0.3
#     ax.set_xlim(minx - dx, maxx + dx)
#     ax.set_ylim(miny - dy, maxy + dy)

#     # Save the plot as a PNG file
#     if save_dir:
#         user_colored_png_file = f"{save_dir}/{value_name}_map.png"
#         plt.savefig(user_colored_png_file, format='png', bbox_inches='tight')
    
#     plt.show()

def save_point_map_with_basemap(df_points, value_name, vmin, vmax, cmap='viridis', marker_size = 15, fill_opacity=0.7, save_dir=None, expand_area_ratio_x = 0.1, expand_area_ratio_y = 0.1, tag=''):
    """
    Creates a scatter plot on a basemap with colors indicating point values.
    Expands the view area around the data points.
    """
    # expand_area_ratio_x = 0.4
    # expand_area_ratio_y = 0.15

    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df_points, geometry=gpd.points_from_xy(df_points.lon, df_points.lat))
    gdf.set_crs("EPSG:4326", inplace=True)  # Set CRS to WGS84
    gdf = gdf.to_crs(epsg=3857)  # Convert to Web Mercator for mapping

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y, c=gdf[value_name], cmap=cmap, alpha=fill_opacity, s=marker_size, vmin=vmin, vmax=vmax)

    # Colorbar
    # plt.colorbar(scatter, ax=ax, label=value_name)

    # Expand the area of the map
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = (maxx - minx) * expand_area_ratio_x
    dy = (maxy - miny) * expand_area_ratio_y
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)

    # Remove axis lines
    ax.set_axis_off()

    # Adding Contextily basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # Save the plot as a PNG file
    if save_dir:
        user_colored_png_file = f"{save_dir}/{value_name}_{tag}_map.png"
        plt.savefig(user_colored_png_file, format='png', bbox_inches='tight', dpi=1000)

    plt.show()

def save_point_map_with_basemap_no_value(df_points, marker_size = 15, fill_color = 'k', fill_opacity=0.7, save_dir=None, expand_area_ratio_x = 0.1, expand_area_ratio_y = 0.1, tag=''):
    """
    Creates a scatter plot on a basemap with colors indicating point values.
    Expands the view area around the data points.
    """
    # expand_area_ratio_x = 0.4
    # expand_area_ratio_y = 0.15

    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(df_points, geometry=gpd.points_from_xy(df_points.lon, df_points.lat))
    gdf.set_crs("EPSG:4326", inplace=True)  # Set CRS to WGS84
    gdf = gdf.to_crs(epsg=3857)  # Convert to Web Mercator for mapping

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y, alpha=fill_opacity, c = fill_color, s=marker_size, edgecolors='none')

    # Colorbar
    # plt.colorbar(scatter, ax=ax, label=value_name)

    # Expand the area of the map
    minx, miny, maxx, maxy = gdf.total_bounds
    dx = (maxx - minx) * expand_area_ratio_x
    dy = (maxy - miny) * expand_area_ratio_y
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)

    # Remove axis lines
    ax.set_axis_off()

    # Adding Contextily basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # Save the plot as a PNG file
    if save_dir:
        user_colored_png_file = f"{save_dir}/points_{tag}_map.png"
        plt.savefig(user_colored_png_file, format='png', bbox_inches='tight')

    plt.show()
