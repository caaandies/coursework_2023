import xarray as xr
import requests
from urllib.request import urlopen
from lxml import etree
import os
import wget
import cdsapi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.sparse.linalg import lsmr
from matplotlib.animation import FuncAnimation
import datetime
from dateutil.relativedelta import relativedelta



file_dict = {
             "daily mslp": "daily_mslp.nc",
             "monthly mslp": "monthly_mslp.nc"
             }

def get_corr(vector1, vector2):
    return np.corrcoef(vector1, vector2)[0][1]

def get_rmse(vector1, vector2):
    return np.sqrt(np.mean((vector1 - vector2) ** 2))

def update_data():
    urls = [
            "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/Dailies/surface/",
            "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/Monthlies/surface/"
            ]
    ds_names = [
                "daily_mslp.nc",
                "monthly_mslp.nc"
                ]
    for index in range(len(urls)):
        file_names = etree.parse(urlopen(urls[index]), etree.HTMLParser()).xpath('//a[contains(@href, "mslp")]/@href')
        
        dataset = None

        for file in sorted(file_names):
            f = wget.download(f"{urls[index]}{file}")
            
            if dataset is None:
                dataset = xr.open_dataset(f)
            else:
                dataset = xr.concat([dataset, xr.open_dataset(f)], dim = 'time')

            os.remove(f)

        dataset.to_netcdf(ds_names[index])


def get_current_state(date, data):
    strdate = date.strftime("%Y-%m-%d")
    if data == "monthly mslp":
        strdate = date.strftime("%Y-%m")
    
    return xr.open_dataset(file_dict[data]).sel(time=strdate)
    

def natural_analogue(date, data, time_units):
    six_months_ago = date - pd.DateOffset(months=6)
    historical_data = xr.open_dataset(file_dict[data]).sel(time=slice(None, six_months_ago))
    historical_mslp = historical_data['mslp'].values
    original_shape = historical_mslp.shape[1]
    historical_mslp = historical_mslp.reshape(historical_mslp.shape[0], -1)

    current_mslp = get_current_state(date, data)['mslp'].values.flatten()

    max_corr = -2
    analog_time = 0;
    rmse = 0;
    for time in range(historical_mslp.shape[0]):
        corr = get_corr(current_mslp, historical_mslp[time])
        if corr > max_corr:
            max_corr = corr
            analog_time = time
    str_time = str(historical_data['time'].values[analog_time])
    rmse = get_rmse(current_mslp, historical_mslp[analog_time])
    return (
            current_mslp.reshape((original_shape, -1)),
            historical_mslp[analog_time + 1: analog_time + time_units + 1].reshape((time_units, original_shape, -1)),
            historical_mslp[analog_time].reshape((original_shape, -1)),
            rmse,
            corr,
            str_time
            )


def calculated_analogue(date, data, time_units):
    six_months_ago = date - pd.DateOffset(months=6)
    historical_data = xr.open_dataset(file_dict[data]).sel(time=slice(None, six_months_ago))
    historical_mslp = historical_data['mslp'].values
    original_shape = historical_mslp.shape[1]
    historical_mslp = historical_mslp.reshape(historical_mslp.shape[0], -1).T

    current_mslp = get_current_state(date, data)['mslp'].values.flatten()

    coeffs = lsmr(historical_mslp, current_mslp)[0]
    analogue = historical_mslp @ coeffs
    rmse = get_rmse(current_mslp, analogue)
    corr = get_corr(current_mslp, analogue)
    return (
            current_mslp.reshape((original_shape, -1)),
            np.array([(historical_mslp[:, i:] @ coeffs[:-i]).reshape((original_shape, -1)) for i in range(1, time_units + 1)]),
            (historical_mslp @ coeffs).reshape((original_shape, -1)),
            rmse,
            corr
            )     



def make_picture(data, title=""):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(title)
    image = ax.imshow(data, origin='upper',
                  extent=[0, 360, -90, 90],
                  transform=ccrs.PlateCarree(),
                  cmap='viridis')
    fig.colorbar(image, ax=ax, orientation='vertical')
    return fig

def make_animation(mslp, date, data):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()

    image = ax.imshow(mslp[0, :, :],
                      extent=[0, 360, -90, 90],
                      transform=ccrs.PlateCarree())
    
    step = datetime.timedelta(days=1)
    format = '%Y-%m-%d'
    
    if (data == 'monthly mslp'):
        step = relativedelta(months=1)
        format = '%Y-%m'
        

    text = ax.text(0.5, 1.01, date.strftime(format),
                   transform=ax.transAxes, ha='center')
    
    def update(index):
        image.set_array(mslp[index, :, :])
        text.set_text((date + (index + 1) * step).strftime(format))
        return image, text
    
    anim = FuncAnimation(fig, update, frames=range(mslp.shape[0]), interval=500, blit=True)
    anim.save('forecast.gif')