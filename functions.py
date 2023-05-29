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
import random
import plotly.graph_objects as go


def get_corr(vector1, vector2):
    return np.corrcoef(vector1, vector2)[0][1]


def get_rmse(vector1, vector2):
    return np.sqrt(np.mean((vector1 - vector2) ** 2))


def update_data(): # upload/update mslp data
    urls = [ # links to data directories
            "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/Dailies/surface/",
            "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/Monthlies/surface/"
            ]
    ds_names = [ # local name of data files
                "daily_mslp.nc",
                "monthly_mslp.nc"
                ]
    for index in range(len(urls)):
        file_names = etree.parse(urlopen(urls[index]), etree.HTMLParser()).xpath('//a[contains(@href, "mslp")]/@href') # getting links to files
        dataset = None
        for file in sorted(file_names):
            f = wget.download(f"{urls[index]}{file}")
            if dataset is None:
                dataset = xr.open_dataset(f)
            else:
                dataset = xr.concat([dataset, xr.open_dataset(f)], dim = 'time')
            os.remove(f)
        dataset.to_netcdf(ds_names[index]) # download the files and combine them into one


class Forecast():
    def __init__(self, data, date, duration, init_state):
        self.data = data # daily_mslp.nc or monthly_mslp.nc
        self.date = date # datetime.date
        self.duration = duration # int > 0
        self.init_state = init_state # netcdf file or None


    def get_current_state(self): 
        if self.init_state is not None: # open the file if it is uploaded
            with NamedTemporaryFile(suffix=".nc") as tmp:
                tmp.write(self.init_state.read())
                tmp.flush()
                return xr.open_dataset(tmp.name)

        if self.data == "daily_mslp.nc": # otherwise, we set the initial state as the state at the time of the entered date
            strdate = self.date.strftime("%Y-%m-%d")
        elif self.data == "monthly_mslp.nc":
            strdate = self.date.strftime("%Y-%m")
        return xr.open_dataset(self.data).sel(time=strdate)


    def natural_analogue(self):
        six_months_ago = self.date - pd.DateOffset(months=6)
        historical_data = xr.open_dataset(self.data).sel(time=slice(None, six_months_ago))
        historical_mslp = historical_data['mslp'].values
        original_shape = historical_mslp.shape[1]
        historical_mslp = historical_mslp.reshape(historical_mslp.shape[0], -1) # getting historical data

        current_mslp = self.get_current_state()['mslp'].values.flatten() # getting initial state

        max_corr = -2
        analog_time = 0;
        for time in range(historical_mslp.shape[0]): # finding the best analog (with the highest correlation)
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
        historical_mslp = historical_mslp.reshape(historical_mslp.shape[0], -1).T # getting historical data

        current_mslp = self.get_current_state()['mslp'].values.flatten() # getting initial state

        coeffs = lsmr(historical_mslp, current_mslp)[0] # finding the coefficients minimizing the norm of the difference of states
        analogue = historical_mslp @ coeffs # getting an analogue
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
        ax.set_title(title) # creating a title
        image = ax.imshow(mslp,
                          extent=[0, 360, -90, 90],
                          transform=ccrs.PlateCarree()) # visualize the data on the map

        fig.colorbar(image, ax=ax, orientation='vertical') # creating a scale
        return fig

    
    def make_animation(self, mslp):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()

        image = ax.imshow(mslp[0, :, :],
                          extent=[0, 360, -90, 90],
                          transform=ccrs.PlateCarree()) # visualize the data (first frame) on the map
        
        if self.data == "daily_mslp.nc":
            step = datetime.timedelta(days=1)
            format = '%Y-%m-%d'
        elif self.data == "monthly_mslp.nc":
            step = pd.DateOffset(months=1)
            format = '%Y-%m' 

        text = ax.text(0.5, 1.01, self.date.strftime(format),
                       transform=ax.transAxes, ha='center')
        fig.colorbar(image, ax=ax, orientation='vertical') # creating a scale
        #first frame created
    
        def update(index): # frame update function
            image.set_array(mslp[index, :, :]) # update data
            text.set_text((self.date + (index + 1) * step).strftime(format)) #update title
            return image, text
    
        anim = FuncAnimation(fig, update, frames=range(mslp.shape[0]), interval=500, blit=True) # creating an animation
        anim.save('forecast.gif') # saving the gif

    
    def test(self, interval, tests_count, method): # to test the accuracy of forecasts

        def random_date(start, end):
            return start + datetime.timedelta(days=random.randint(0, (end - start).days))

        corr_array = []
        rmse_array = []
        for duration in range(interval[0], interval[1] + 1):
            self.duration = duration
            corr_array.append(0)
            rmse_array.append(0)
            for attemp in range(tests_count):
                self.date = random_date(datetime.date(2001, 1, 1), datetime.date(2021, 12, 31))
                if method == "Natural analogue":
                    prediction = (self.natural_analogue())[1][-1]
                elif method == "Calculated analogue":
                    prediction = (self.calculated_analogue())[1][-1]
                # got a forecast
                if self.data == "daily_mslp.nc":
                    strdate = self.date.strftime("%Y-%m-%d")
                elif self.data == "monthly_mslp.nc":
                    strdate = self.date.strftime("%Y-%m")
                real = xr.open_dataset(self.data).sel(time=strdate)['mslp'].values
                # got what it really was
                corr_array[-1] += get_corr(real.flatten(), prediction.flatten()) # getting correlation
                rmse_array[-1] += get_rmse(real.flatten(), prediction.flatten()) #getting Euclidean distance
            corr_array[-1] /= tests_count
            rmse_array[-1] /= tests_count
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=list(range(interval[0], interval[1] + 1)), y=corr_array))
        fig1.update_layout(xaxis_title="duration", yaxis_title="correlation", title_text="Correlation graphic")
        # made a correlation graph from the duration of the forecast
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(range(interval[0], interval[1] + 1)), y=rmse_array, name='rmse'))
        fig2.update_layout(xaxis_title="duration", yaxis_title="rmse", title_text="RMSE graphic")
        # made a Euclidean distance graph from the duration of the forecast
        return (fig1, fig2)

