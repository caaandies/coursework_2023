import xarray as xr
from urllib.request import urlopen
from lxml import etree
import os
import wget
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.sparse.linalg import lsmr
from matplotlib.animation import FuncAnimation
import datetime
from tempfile import NamedTemporaryFile


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


class Forecast():
    def __init__(self, data, date, duration, init_state):
        self.data = data # daily_mslp.nc or monthly_mslp.nc
        self.date = date # datetime.date
        self.duration = duration # int > 0
        self.init_state = init_state # netcdf file or None
    
    def get_current_state(self):
        if self.init_state is not None:
            with NamedTemporaryFile() as tmp:
                tmp.write(self.init_state.get_value())
                return xr.open_dataset(tmp.name)
        if self.data == "daily_mslp.nc":
            strdate = self.date.strftime("%Y-%m-%d")
        elif self.data == "monthly_mslp.nc":
            strdate = self.date.strftime("%Y-%m")
        return xr.open_dataset(self.data).sel(time=strdate)


    def natural_analogue(self):
        six_months_ago = self.date - pd.DateOffset(months=6)
        historical_data = xr.open_dataset(self.data).sel(time=slice(None, six_months_ago))
        historical_mslp = historical_data['mslp'].values
        original_shape = historical_mslp.shape[1]
        historical_mslp = historical_mslp.reshape(historical_mslp.shape[0], -1)

        current_mslp = self.get_current_state()['mslp'].values.flatten()

        max_corr = -2
        analog_time = 0;
        for time in range(historical_mslp.shape[0]):
            corr = get_corr(current_mslp, historical_mslp[time])
            if corr > max_corr:
                max_corr = corr
                analog_time = time
        analogue_time = pd.to_datetime(str(historical_data['time'].values[analog_time]))
        if self.data == "daily_mslp.nc":
            analogue_time = analogue_time.strftime('%Y-%m-%d')
        elif self.data == "monthly_mslp.nc":
            analogue_time = analogue_time.strftime('%Y-%m')
        return (
                current_mslp.reshape((original_shape, -1)),
                historical_mslp[analog_time + 1: analog_time + self.duration + 1].reshape((self.duration, original_shape, -1)),
                historical_mslp[analog_time].reshape((original_shape, -1)),
                get_rmse(current_mslp, historical_mslp[analog_time]),
                max_corr,
                analogue_time
                )



    def calculated_analogue(self):
        six_months_ago = self.date - pd.DateOffset(months=6)
        historical_data = xr.open_dataset(self.data).sel(time=slice(None, six_months_ago))
        historical_mslp = historical_data['mslp'].values
        original_shape = historical_mslp.shape[1]
        historical_mslp = historical_mslp.reshape(historical_mslp.shape[0], -1).T

        current_mslp = self.get_current_state()['mslp'].values.flatten()

        coeffs = lsmr(historical_mslp, current_mslp)[0]
        analogue = historical_mslp @ coeffs
        return (
                current_mslp.reshape((original_shape, -1)),
                np.array([(historical_mslp[:, i:] @ coeffs[:-i]).reshape((original_shape, -1)) for i in range(1, self.duration + 1)]),
                analogue.reshape((original_shape, -1)),
                get_rmse(current_mslp, analogue),
                get_corr(current_mslp, analogue)
                )



    def make_picture(self, mslp, title=""):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_title(title)
        image = ax.imshow(mslp,
                          extent=[0, 360, -90, 90],
                          transform=ccrs.PlateCarree())
        fig.colorbar(image, ax=ax, orientation='vertical')
        return fig
    
    def make_animation(self, mslp):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()

        image = ax.imshow(mslp[0, :, :],
                          extent=[0, 360, -90, 90],
                          transform=ccrs.PlateCarree())
        
        if self.data == "daily_mslp.nc":
            step = datetime.timedelta(days=1)
            format = '%Y-%m-%d'
        elif self.data == "monthly_mslp.nc":
            # step = relativedelta(months=1)
            step = pd.DateOffset(months=1)
            format = '%Y-%m' 

        text = ax.text(0.5, 1.01, self.date.strftime(format),
                       transform=ax.transAxes, ha='center')
        fig.colorbar(image, ax=ax, orientation='vertical')
    
        def update(index):
            image.set_array(mslp[index, :, :])
            text.set_text((self.date + (index + 1) * step).strftime(format))
            return image, text
    
        anim = FuncAnimation(fig, update, frames=range(mslp.shape[0]), interval=500, blit=True)
        anim.save('forecast.gif')
