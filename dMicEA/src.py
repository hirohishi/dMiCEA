import os
import trackpy as tp
import pandas as pd
import copy
from scipy.ndimage import gaussian_filter


import matplotlib.pyplot as plt
import sys
import numpy as np
import click
from scipy.spatial import distance
import math
import numpy as np
from nd2reader import ND2Reader
import trackpy as tp
import pandas as pd
from skimage.morphology import white_tophat, black_tophat, disk
from read_roi import read_roi_zip
import seaborn as sns
from scipy.ndimage import median_filter
from skimage import restoration
import cv2
#import tensorflow as tf
import deepblink as pink
from cellpose import models, core, io
use_GPU = core.use_gpu()

import warnings
warnings.simplefilter('ignore')

tp.quiet()


@click.group()
def cli():
    """This package can be used to analyze gene regulation in micro enviroment in cells.The analysis file contains 3 color images, first is transcription factors staining, second is MCP (nascent transcription), third is mTetR (gene locus). Image processing using scipy module for nd2 file. Determine transctiptional state by MCP spots intensity and distances between MCP and mTetR. """
    pass


@cli.command()
@click.option("--nd2_file", "-f1", help="File path of Nd2 file", type=str, required=True)
@click.option("--csv_file","-f2", help="File path of mTetR spot csv data", type=str, required=True)
@click.option("--dia_spot", default=5, help="Estimated spot diameter, more than 5 and odd number", type=int, required=False)
@click.option("--gauss", default=5, help="Gaussian fitting parameter, more than 2", type=int, required=False)
@click.option("--out_path","-out", default='./MicEA_data_output', help="Output file path, default is ./MicEA_data_output", type=str, required=False)
@click.option("--pixel_size","-ps", default=130, help="Pixel size of image, default is 130.", type=int, required=False)
@click.option("--roi_size","-rs", default=(19, 19), help="ROI size for analysis, default is 19x19. Put y x values like -rs 19 19.", type=(int, int), required=False)
def analysis(nd2_file, csv_file, dia_spot, gauss, out_path, pixel_size, roi_size):

    """2nd: Analyze distance between protein cluster and annotated mTetR spots (gene locus)."""
    nd2_file_path, roi_file_path, diameter_1, diameter_2, out_path = nd2_file, csv_file, dia_spot, gauss, out_path
    y_rad = int(roi_size[0]/2)
    x_rad = int(roi_size[1]/2)
    #make output folder
    os.makedirs(out_path, exist_ok=True)

    print ("Start of image processing and analysis")

    print('nd2 file path:', nd2_file_path)
    print('spot position file path:', roi_file_path)
    print('Spot diameter:', diameter_1) # more than 5 and odd number
    print('Gaussian fitting parameter:', diameter_2) # more than 2
    
    file_name = nd2_file_path.split("/")

    #Get information from the spots file.
    if('.zip' in roi_file_path ):
        roi = read_roi_zip(roi_file_path)
        df_anno = func_roi_to_df(roi)
    elif('.csv' in roi_file_path):
        df_anno = pd.read_csv(roi_file_path, index_col=0)
    else:
        print ("Please set the roi or csv file.")
        sys.exit()

    if(len(df_anno) == 0):
        print ("No mTetR spots in this nd2 file.")
        sys.exit()

    image_list = []

    df_3_all = pd.DataFrame()

    print ("Image processing.")

    #Loop through the number of colors.
    for color in range(3):
        #Get images from the nd2 file.
        img_o = func_read_noFOV(nd2_file_path, 't', color)
        img_o = gaussian_filter(img_o, sigma = 1) #Apply a Gaussian filter to the image.
        if(color == 1 or color == 2): 
            output = [subtract_background(img_o[i], radius=5) for i in range(10)]
        else:
            output = [subtract_background(img_o[i], radius=5) for i in range(10)]

        OUTPUT = np.array(output).mean(axis=0)

        image_list.append(OUTPUT) #Add images to image_list.

    #Give each spot an id in case there are multiple ROIs in each image.
    spot_id = 0

    #Loop through the number of ROIs in each image.
    for row in df_anno.itertuples():
        spot_id = spot_id + 1
        #Get a 19x19 pixel image centered on each ROI coordinate in each image.
        y_1 = int(row[1]) - y_rad
        y_2 = int(row[1]) + y_rad + 1
        x_1 = int(row[2]) - x_rad
        x_2 = int(row[2]) + x_rad + 1

        image_list_c = copy.copy(image_list)

        image_1 = image_list_c[0][y_1:y_2, x_1:x_2]
        image_2 = image_list_c[1][y_1:y_2, x_1:x_2]
        image_3 = image_list_c[2][y_1:y_2, x_1:x_2]

        print (f"Spot ID: {spot_id}, Detection of the center of spots in all images and calculation of 2D distance and transcriptional state.")
        #Detect the center of spots in the SNAPtag image
        df_1 = tp.locate(image_1, diameter_1)
        df_1 = tp.refine_leastsq(df_1, image_1, diameter_2, fit_function='gauss')
        df_1 = df_1.reset_index(drop=True)
        df_posi_1 = df_1.loc[:, ['y', 'x']]
        df_int_1 = func_intensity(df_posi_1.values, image_1)
        df_1['r_snap'] = df_int_1['r_mass']
        #print (df_1)

        #Detect the center of spots in the MCP image.
        df_2 = tp.locate(image_2, diameter_1, topn=1)
        df_2 = tp.refine_leastsq(df_2, image_2, diameter_2, fit_function='gauss')
        df_2 = df_2.reset_index(drop=True)
        df_posi_2 = df_2.loc[:, ['y', 'x']]
        df_int_2 = func_intensity(df_posi_2.values, image_2)
        df_2['r_mcp'] = df_int_2['r_mass']
        #print (df_2)

        #Detect the center of spots in the mTetR image.
        df_3 = tp.locate(image_3, diameter_1, topn=1)
        df_3 = tp.refine_leastsq(df_3, image_3, diameter_2, fit_function='gauss')
        df_3 = df_3.reset_index(drop=True)
        df_posi_3 = df_3.loc[:, ['y', 'x']]
        df_int_3 = func_intensity(df_posi_3.values, image_3)
        df_3['r_tetr'] = df_int_3['r_mass']
        #print (df_int_3)
        #print (df_3)

        #Calculate the distance between the mTetR and SNAPtag spots.
        index, dist_s = func_dis_min_2d(df_3['x'], df_3['y'], df_1['x'], df_1['y'])
        #print (dist_s)

        #Add data to df_3.
        df_3['dist_s_t'] = dist_s * 130 #Convert the unit of the distance between the mTetR and SNAPtag spots to nm.
        df_3['r_mcp'] = df_2['r_mcp'].values
        df_3['y_mcp'] = df_2['y'].values
        df_3['x_mcp'] = df_2['x'].values
        df_3['r_snap'] = df_1.iloc[index, -1]
        df_3['raw_snap'] = df_1.iloc[index, -2]
        df_3['y_snap'] = df_1.iloc[index, 0]
        df_3['x_snap'] = df_1.iloc[index, 1]
        df_3['file_name'] = nd2_file_path
        df_3['spot_id'] = spot_id
        df_3['y_raw'] = row[1]
        df_3['x_raw'] = row[2]

        #Judge whether the MCP spot is ON or OFF state.
        if (len(df_2) > 0):
            index_m, dist_m = func_dis_min_2d(df_3['x'], df_3['y'], df_2['x'], df_2['y'])
            df_3['dist_m_t'] = dist_m * pixel_size
            if (df_2['r_mcp'].values >= 2 and dist_m <= 3):
                df_3['state'] = 'ON'
            else:
                df_3['state'] = 'OFF'
        else:
            df_3['state'] = 'OFF'

        #Add data to df_3_all.
        df_3_all = df_3_all.append(df_3)

        plot_data_save(file_name, out_path, spot_id, df_1, df_2, df_3, image_1, image_2, image_3)

    df_3_all.to_csv(f"{out_path}/MicEA_{file_name[-1]}.csv")

    print ("Done.")
    
    return


def plot_data_save(file_name, out_path, spot_id, df_1, df_2, df_3, image_1, image_2, image_3):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)  
    ax3 = fig.add_subplot(143)  
    ax4 = fig.add_subplot(144)  

    tp.annotate(df_1, image_1, ax = ax1)
    tp.annotate(df_2, image_2, ax = ax2)
    tp.annotate(df_3, image_3, ax = ax3)
    #Set the font in the graph.
    #plt.rcParams["font.family"] = "Arial"

    size_im = image_1.shape

    ax4.plot([df_3['x'].values, df_3['x_snap'].values], [df_3['y'].values, df_3['y_snap'].values], color="black", zorder=1)
    ax4.scatter(df_3['x'], df_3['y'], zorder=2, label="mTetR")
    ax4.scatter(df_1['x'], df_1['y'], zorder=2, label="Cluster")
    if(df_3['state'].values == "ON"):
        ax4.scatter(df_2['x'], df_2['y'], zorder=2, label="MCP")

    ax4.set_ylim(0, size_im[0])
    ax4.set_xlim(0, size_im[1])
    ax4.invert_yaxis()
    ax4.set_aspect('equal', adjustable='box')

    ax4.set_xticks(np.linspace(0, size_im[0]-1, size_im[0]))
    #ax4.set_xticks(np.linspace(0, 18, 19), minor=True)
    ax4.set_yticks(np.linspace(0, size_im[0]-1, size_im[0]))
    #ax4.set_yticks(np.linspace(0, 18, 19), minor=True)
    ax4.grid(alpha = 0.3)
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))

    ax1.set_title('Protein Cluster')
    ax2.set_title('MCP')
    ax3.set_title('mTetR')
    ax4.set_title(f"{df_3['dist_s_t'].values[0]}")

    plt.suptitle(f"{file_name[-1]}, spot_id: {str(spot_id)},  state: {df_3['state'].values[0]}")
    #plt.show()
    plt.tight_layout()
    #plt.savefig(f"./220729-output_{spot_id}.pdf")
    plt.savefig(f"{out_path}/MicEA_{file_name[-1]}_spotID_{spot_id}.png")

    return

def func_distance_2d(x_1, y_1, x_2, y_2):  # all are values
    dis = distance.euclidean((x_1, y_1), (x_2, y_2))
    
    return dis


def func_dis_min_2d(a, b, c, d):  # a and b are values, c, d are list or df
    dis_spot = []
    a = float(a.values)
    b = float(b.values)
    c = c.values
    d = d.values\
    #print (a, b, c, d)
    for x, y in zip(c, d):
        dis = distance.euclidean([a, b], [float(x), float(y)])
        dis_spot.append(dis)
    aa = dis_spot.index(min(dis_spot)) #index
    bb = min(dis_spot) #distance
    
    return aa, bb

def func_intensity(yx, target_image):
    mean_value = target_image.mean()
    df = pd.DataFrame()
    pad_width = radius_refinement = 3
    y, x = yx.T.copy()
    yx = yx + pad_width
    target_image = np.pad(
        target_image, pad_width=pad_width, mode='constant', constant_values=0
    )

    df_curr = tp.refine_com(
        raw_image=target_image, image=target_image, radius=radius_refinement, coords=yx
    )

    df_curr["x"] = x
    df_curr["y"] = y

    df = df.append(df_curr, ignore_index=True)
    df['mean_back_int'] = mean_value
    df['mean_mass'] = df['mass'] / (3 * 3 * math.pi)
    df['r_mass'] = df['mean_mass'] / df['mean_back_int']

    return df

def func_read_noFOV(path, iter, col):
    img = ND2Reader(path)
    img.bundle_axes = 'yx'
    img.iter_axes = iter
    img.default_coords['c'] = col
    
    return img

def func_roi_to_df(roi):
    df_anno = pd.DataFrame()
    for name, ROI in roi.items():
        df_roi = pd.DataFrame(data =[[ROI['y'][0],ROI['x'][0]]], columns = ['y', 'x'])
        df_anno = pd.concat([df_anno, df_roi])

    return df_anno

def subtract_background(image, radius=50, light_bg=False):
    str_el = disk(radius)
    if light_bg:
        return black_tophat(image, str_el)
    else:
        return white_tophat(image, str_el)

@cli.command()
@click.option("--csv_folder_path","-cp", help="Folder path of output data (csv)", type=str, required=True)
@click.option("--out_path","-out", default='./MicEA_data_output', help="Output file path, default is ./MicEA_data_output", type=str, required=False)
def collect(csv_folder_path, out_path):

    """Combine output csv file from MicEA analysis command."""

    output_folder = os.path.join(out_path, "4_data_analysis")
    os.makedirs(output_folder, exist_ok=True)

    files = next(os.walk(csv_folder_path))[2]
    df = pd.DataFrame()
    for file in files:
        if('.nd2.csv' in file):
            df_each = pd.read_csv(os.path.join(csv_folder_path, file))
            if(len(df_each) > 0):
                df = pd.concat([df, df_each])

    df.to_csv(os.path.join(output_folder, f"Data_Comb_MicEA.csv"))
    return


@cli.command()
@click.option("--csv_file_path","-cp", help="File path of output data (csv)", type=str, required=True)
@click.option("--target","-ta", help="What is your target including Cluster or MCP??", type=str, required=True)
@click.option("--out_path","-out", default='./MicEA_data_output', help="Output file path, default is ./MicEA_data_output", type=str, required=False)
@click.option("--type","-ty", help="Please select plot type including bp(boxplot), sp(scatter plot), cdf(cdf plot)", type=str, required=True)
@click.option("--pixel_size","-ps", default=130, help="Pixel size of image, default is 130.", type=int, required=False)
def plot(csv_file_path, out_path, type, target, pixel_size):

    """3rd: Plot csv file from MicEA plot command."""

    output_folder = os.path.join(out_path, "4_data_analysis")

    df = pd.read_csv(os.path.join(csv_file_path))
    if(type == "bp"):
        boxplot(df, output_folder, target)
    elif(type == "sp"):
        scatter(df, output_folder, target, pixel_size)
        print ("developping")
    elif(type == "cdf"):
        print ("developping")
    else:
        print ("Select correct plot type.")
        sys.exit()

    return


def boxplot(df, out_path, target):
    sns.set_theme()

    if(target == "Cluster"):
        y = 'dist_s_t'
    elif(target == "MCP"):
        y = 'dist_m_t'

    x = "state"

    ax = sns.boxplot(data=df, x=x, y=y, showfliers=False)
    sns.stripplot(x=x, y=y, data=df, jitter=True, color='black', ax=ax, alpha=0.2)
    ax.set_ylabel('2D distance (nm)')
    #plt.show()
    plt.savefig(f"{out_path}/Boxplot_MicEA_2D_mTetR_{target}.png")

    return


def scatter(df, out_path, target, pixel_size):

    if(target == "Cluster"):
        target_1 = 'snap'
    elif(target == "MCP"):
        target_1 = 'mcp'

    x = "state"

    #Calculate the relative coordinate of SNAPtag spots against mTetR spots.
    df['y_t.s'] = df[f'y_{target_1}'] - df['y']
    df['x_t.s'] = df[f'x_{target_1}'] - df['x']

    ##Convert relative coordinate to µm.
    df['y_t.s_µm'] = df['y_t.s'] * pixel_size/1000
    df['x_t.s_µm'] = df['x_t.s'] * pixel_size/1000

    sns.set_theme()

    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(1, 1, 1)

    ##Draw scatter plot using seaborn.
    sns.scatterplot(x = 'x_t.s_µm', y = 'y_t.s_µm', data = df, hue = x, ax=ax)
    plt.ylim(-1.2,1.2) #Set the range of y axis.
    plt.xlim(-1.2,1.2) #Set the range of x axis.

    ax.set_xlabel('Relative x coordinate (µm)')
    ax.set_ylabel('Relative y coordinate (µm)')
    #plt.show()
    plt.savefig(f"{out_path}/Scatterplot_MicEA_2D_mTetR_{target}.png")

    return


def seg_id(file_path, out_path):
    color = 0
    img_o = func_read_noFOV(file_path, 't', color)

    TEST, img_list_A = func_image_AIP(img_o, 10) #img_oのAIP
    test2 = median_filter(TEST[0], 10)

    model = models.Cellpose(gpu=use_GPU, model_type='nuclei')
    masks, flows, styles, diams = model.eval(test2, diameter = 100)

    io.save_to_png(test2, masks, flows, f"{out_path}/segmentation.png")

    filename= f"{out_path}/segmentation_cp_masks.png"
    img = io.imread(filename)
    
    size = img.shape
    
    edge = np.array([img[0], img[size[0]-1], img[:,0], img[:,size[1]-1]])
    edge_list = np.unique(edge[edge > 0]) 

    for i in edge_list:
        img[img == i] = 0

    cell_id = np.unique(img[img > 0])

    return img, cell_id


def load_model(model_path):
    model_tetr = pink.io.load_model(model_path) # "/home/ohishi/Bioim/data_folder/ak_1_tetr.h5"
    return model_tetr


def fix_spot(df):
    cell_sep = df['cell_id'].unique()
    cell_sep = cell_sep[cell_sep != 0]
    df_anno = pd.DataFrame()
    for cell in cell_sep:
        df_id = df[df['cell_id']== cell]
        maxmass = df_id['mass'].max()
        #print (maxmass)
        df_id_2 = df_id[df_id['mass'] == maxmass]
        df_anno = df_anno.append(df_id_2)
    return df_anno


def deep_local_TetR(file_path, img, model_tetr):
    color = 2  
    img_o = func_read_noFOV(file_path, 't', color)
    IMG_A, img_a = func_image_AIP(img_o, 10)
    IMG_A = gaussian_filter(IMG_A[0], sigma=1)
    background = restoration.rolling_ball(IMG_A, radius=50)
    output = IMG_A - background
    yx = pink.inference.predict(image=output, model=model_tetr)
    df = func_intensity(yx, output)

    c_id_list = []
    for y, x in zip(df['y'], df['x']):
        y = int(y)
        x = int(x)
        c_id = img[y,x]
        c_id_list.append(c_id)

    df['cell_id'] = c_id_list
    
    return df


def func_image_AIP(img, z_stack):
    img_list = [np.fft.fftn(img[i]) for i in range(z_stack)]
    img_list = [np.fft.ifftn(img_list[i]) for i in range(z_stack)]
    img_list = [np.uint16(img_list[i]) for i in range(z_stack)]
    IMG_xy = np.mean(img_list, axis=0) #xy
    IMG_zx = np.mean(img_list, axis=1) #zx
    IMG_zy = np.mean(img_list, axis=2) #zy
    IMG_LIST = [IMG_xy, IMG_zx, IMG_zy]

    return IMG_LIST, img_list


@cli.command()
@click.option("--input_folder","-if", help="Folder path of nd2 file.", type=str, required=True)
@click.option("--out_path","-out", default='./MicEA_data_output', help="Output file path, default is ./MicEA_data_output", type=str, required=False)
@click.option("--model","-m", help="Please select your model made by deepblink", type=str, required=True)
def deefind(input_folder, out_path, model):
    """1st: Analyze Segmentation of nucleus using cellpose and mTetR spots candidate using deepblink"""
    onlyfiles = next(os.walk(input_folder))[2]

    output_folder = out_path
    output_seg_folder = f"{out_path}/1_segmentation/"
    output_spot_folder = f"{out_path}/2_spot_detection/" 

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_seg_folder, exist_ok=True)
    os.makedirs(output_spot_folder, exist_ok=True)

    for file in onlyfiles:
        print (file)
        if('.nd2' in file):
            img, cell_id = seg_id(os.path.join(input_folder, file), out_path)
            cv2.imwrite(os.path.join(output_seg_folder, f"{file}_seg.tif"), img)
            model_tetr = load_model(model)
            df = deep_local_TetR(os.path.join(input_folder, file), img, model_tetr)
            df_anno = fix_spot(df)
            df_anno.to_csv(os.path.join(output_spot_folder, f"{file}_dspot.csv"))
    
    return

@cli.command()
@click.option("--input_folder","-if", help="Folder path of nd2 file.", type=str, required=True)
@click.option("--csv_folder","-f2", help="Folder path of mTetR spot csv data", type=str, required=True)
@click.option("--dia_spot", default=5, help="Estimated spot diameter, more than 5 and odd number", type=int, required=False)
@click.option("--gauss", default=5, help="Gaussian fitting parameter, more than 2", type=int, required=False)
@click.option("--out_path","-out", default='./MicEA_data_output', help="Output file path, default is ./MicEA_data_output", type=str, required=False)
@click.option("--pixel_size","-ps", default=130, help="Pixel size of image, default is 130.", type=int, required=False)
@click.option("--roi_size","-rs", default=(19, 19), help="ROI size for analysis, default is 19x19. Put y x values like -rs 19 19.", type=(int, int), required=False)
def analyses(input_folder, csv_folder, dia_spot, gauss, out_path, pixel_size, roi_size):

    """2nd: For many files analysis similar to command of analysis. Analyze distance between protein cluster and annotated mTetR spots (gene locus). For all file. Please set nd2 file folder for analysis."""
    print ("Start of image processing and analysis")

    print('nd2 folder path:', input_folder)
    print('spot position folder path:', csv_folder)
    print('Spot diameter:', dia_spot) # more than 5 and odd number
    print('Gaussian fitting parameter:', gauss) # more than 2

    onlyfiles = next(os.walk(input_folder))[2]
    output_spot_folder2 =  f"{out_path}/3_spot_analysis/" 
    os.makedirs(output_spot_folder2, exist_ok=True)

    for file in onlyfiles:
        print (file)
        if('.nd2' in file):
            nd2_file = os.path.join(input_folder, file)
            csv_file = os.path.join(csv_folder, f"{file}_dspot.csv" )
            analysis2(nd2_file, csv_file, dia_spot, gauss, output_spot_folder2, pixel_size, roi_size)
    
    return

def analysis2(nd2_file, csv_file, dia_spot, gauss, out_path, pixel_size, roi_size):

    """2nd: Analyze distance between protein cluster and annotated mTetR spots (gene locus)."""
    nd2_file_path, roi_file_path, diameter_1, diameter_2, out_path = nd2_file, csv_file, dia_spot, gauss, out_path
    y_rad = int(roi_size[0]/2)
    x_rad = int(roi_size[1]/2)    
    file_name = nd2_file_path.split("/")

    #Get information from the spots file.
    if('.zip' in roi_file_path ):
        roi = read_roi_zip(roi_file_path)
        df_anno = func_roi_to_df(roi)
    elif('.csv' in roi_file_path):
        df_anno = pd.read_csv(roi_file_path, index_col=0)
    else:
        print ("Please set the roi or csv file.")
        return

    if(len(df_anno) == 0):
        print ("No mTetR spots in this nd2 file.")
        return

    image_list = []

    df_3_all = pd.DataFrame()

    #Loop through the number of colors.
    for color in range(3):
        #Get images from the nd2 file.
        img_o = func_read_noFOV(nd2_file_path, 't', color)
        img_o = gaussian_filter(img_o, sigma = 1) #Apply a Gaussian filter to the image.
        if(color == 1 or color == 2): 
            output = [subtract_background(img_o[i], radius=5) for i in range(10)]
        else:
            output = [subtract_background(img_o[i], radius=5) for i in range(10)]

        OUTPUT = np.array(output).mean(axis=0)

        image_list.append(OUTPUT) #Add images to image_list.

    #Give each spot an id in case there are multiple ROIs in each image.
    spot_id = 0

    #Loop through the number of ROIs in each image.
    for row in df_anno.itertuples():
        spot_id = spot_id + 1
        #Get a 19x19 pixel image centered on each ROI coordinate in each image.
        y_1 = int(row[1]) - y_rad
        y_2 = int(row[1]) + y_rad + 1
        x_1 = int(row[2]) - x_rad
        x_2 = int(row[2]) + x_rad + 1

        image_list_c = copy.copy(image_list)

        image_1 = image_list_c[0][y_1:y_2, x_1:x_2]
        image_2 = image_list_c[1][y_1:y_2, x_1:x_2]
        image_3 = image_list_c[2][y_1:y_2, x_1:x_2]

        print (f"Spot ID: {spot_id}, Detection of the center of spots in all images and calculation of 2D distance and transcriptional state.")
        #Detect the center of spots in the SNAPtag image
        df_1 = tp.locate(image_1, diameter_1)
        df_1 = tp.refine_leastsq(df_1, image_1, diameter_2, fit_function='gauss')
        df_1 = df_1.reset_index(drop=True)
        df_posi_1 = df_1.loc[:, ['y', 'x']]
        df_int_1 = func_intensity(df_posi_1.values, image_1)
        df_1['mean_snap'] = df_int_1['mean_mass']
        df_1['r_snap'] = df_int_1['r_mass']
        #print (df_1)

        #Detect the center of spots in the MCP image.
        df_2 = tp.locate(image_2, diameter_1, topn=1)
        df_2 = tp.refine_leastsq(df_2, image_2, diameter_2, fit_function='gauss')
        df_2 = df_2.reset_index(drop=True)
        df_posi_2 = df_2.loc[:, ['y', 'x']]
        df_int_2 = func_intensity(df_posi_2.values, image_2)
        df_2['r_mcp'] = df_int_2['r_mass']
        #print (df_2)

        #Detect the center of spots in the mTetR image.
        df_3 = tp.locate(image_3, diameter_1, topn=1)
        df_3 = tp.refine_leastsq(df_3, image_3, diameter_2, fit_function='gauss')
        df_3 = df_3.reset_index(drop=True)
        df_posi_3 = df_3.loc[:, ['y', 'x']]
        df_int_3 = func_intensity(df_posi_3.values, image_3)
        df_3['r_tetr'] = df_int_3['r_mass']

        df_int_4 = func_intensity(df_posi_3.values, image_1)
        df_3['mean_snap_tetr'] = df_int_4['mean_mass']
        df_3['r_snap_tetr'] = df_int_4['r_mass']
        df_3['mean_snap_back_int'] = df_int_4['mean_back_int']
        #print (df_int_3)
        #print (df_3)

        #Calculate the distance between the mTetR and SNAPtag spots.
        index, dist_s = func_dis_min_2d(df_3['x'], df_3['y'], df_1['x'], df_1['y'])
        #print (dist_s)

        #Add data to df_3.
        df_3['dist_s_t'] = dist_s * 130 #Convert the unit of the distance between the mTetR and SNAPtag spots to nm.
        df_3['r_mcp'] = df_2['r_mcp'].values
        df_3['y_mcp'] = df_2['y'].values
        df_3['x_mcp'] = df_2['x'].values
        df_3['r_snap'] = df_1.iloc[index, -1]
        df_3['mean_snap'] = df_1.iloc[index, -2]
        df_3['y_snap'] = df_1.iloc[index, 0]
        df_3['x_snap'] = df_1.iloc[index, 1]
        df_3['file_name'] = nd2_file_path
        df_3['spot_id'] = spot_id
        df_3['y_raw'] = row[1]
        df_3['x_raw'] = row[2]

        #Judge whether the MCP spot is ON or OFF state.
        if (len(df_2) > 0):
            index_m, dist_m = func_dis_min_2d(df_3['x'], df_3['y'], df_2['x'], df_2['y'])
            df_3['dist_m_t'] = dist_m * pixel_size
            if (df_2['r_mcp'].values >= 2 and dist_m <= 3):
                df_3['state'] = 'ON'
            else:
                df_3['state'] = 'OFF'
        else:
            df_3['state'] = 'OFF'

        #Add data to df_3_all.
        df_3_all = df_3_all.append(df_3)

        plot_data_save(file_name, out_path, spot_id, df_1, df_2, df_3, image_1, image_2, image_3)

    df_3_all.to_csv(f"{out_path}/MicEA_{file_name[-1]}.csv")

    print ("Done.")
    
    return


def meanspot(nd2_folder, csv_folder, ):

    return

if __name__ == "__main__": 
    cli()