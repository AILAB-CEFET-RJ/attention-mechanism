import xarray as xr
import matplotlib.pyplot as plt
import os

#files = os.path.join('../test/*.nc')
#files = os.path.join('../validation/*.nc')
files = os.path.join('../training/*.nc')
ds = xr.open_mfdataset(files, engine="netcdf4")

reflectivity_ds = ds.equivalent_reflectivity_factor  

x_data_s = reflectivity_ds

x = list(set(x_data_s.coords['x'].values.astype('int16')))
y = list(set(x_data_s.coords['y'].values.astype('int16')))
new_y = list(range(y[0],y[-1],205))
new_x = list(range(x[0],x[-1],205))

reflectivity_dsw = x_data_s.interp(x=new_x, y=new_y, method='nearest')

def window_generator_xarray(dataset, n_steps):
    x, y = list(), list()
    for i in range(len(dataset)):
        end_ix = i + n_steps
        end_iy = end_ix + n_steps

        x_data = dataset.isel(time=slice(i,end_ix))
        x_data = x_data.expand_dims(dim='sample', axis=0)
        y_data = dataset.isel(time=slice(end_ix,end_iy))
        y_data = y_data.expand_dims(dim='sample', axis=0)

        if y_data.time.size < n_steps:
            break
        
        x_data = x_data.drop(labels='time')
        x.append(x_data)
        y_data = y_data.drop(labels='time')
        y.append(y_data)

    return x, y

x_list, y_list = window_generator_xarray(reflectivity_dsw, n_steps=5)
x_data = xr.concat(x_list, dim='sample')
x_data = x_data.expand_dims(dim='channel', axis=-1)

y_data = xr.concat(y_list, dim='sample')
y_data = y_data.expand_dims(dim='channel', axis=-1)

print('Sample size with X: ', x_data.sample.size)
print('Sample size with y: ', y_data.sample.size)

print(x_data)

print(x_data.shape)

x_data.transpose("sample","channel","time","x","y").to_netcdf("../dataset/training.nc")