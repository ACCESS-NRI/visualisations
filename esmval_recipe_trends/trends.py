"""diagnostic script to plot"""

import iris
import os
import logging
import numpy as np
import xarray as xr

from esmvaltool.diag_scripts.shared import (
    run_diagnostic,
    save_figure,
    group_metadata,
    select_metadata,
)
from esmvalcore.preprocessor import (
    climate_statistics,
    regrid,
    concatenate
)


# This part sends debug statements to stdout
logger = logging.getLogger(os.path.basename(__file__))


def calc_trend(xarr, start, end, length):
    """
    Compute running linear trends over windows of size `length` (years)
    from `start` to `end - 1`, returning trend per decade.

    xarr must have a coordinate "year".
    """
    
    trend_list = []

    for yr in range(start, end):
        # window
        win = xarr.sel(year=slice(str(yr), str(yr + length-1)))

        # replace year coordinate with integers 0..N-1 for polyfit
        t = np.arange(win.year.size)
        win = win.assign_coords(year=t)

        # polyfit: slope * 10 for trend per decade
        tr = (win.polyfit(dim="year", deg=1, skipna=True)
                 .polyfit_coefficients
                 .isel(degree=0) * 10)
        
        trend_list.append(tr)

    # concatenate trends along a new "year" dimension
    trnarr = xr.concat(trend_list, dim="year")

    # restore actual years
    trnarr = trnarr.assign_coords(year=np.arange(start, end))

    return trnarr


def main(cfg):
    """run on all datasets"""

    input_data = cfg["input_data"].values()
    ## group by dataset to concatenate future to historical after ensemble means
    # for dataset, attributes in group_metadata(input_data,"dataset", sort='exp').items():
    #     in_files = [d['filename'] for d in attributes]
    #     cubes = iris.load(in_files)
    #     start_year = min([d['start_year'] for d in attributes]) 
    #     end_year = max([d['end_year'] for d in attributes])
        
    #     logger.info(f"Loaded {len(cubes)} cubes for {dataset}, {start_year} - {end_year}")
    #     iris.util.equalise_attributes(cubes)
    #     cube = concatenate(cubes)

    ## keep future variable separate 
    for dataset in input_data:
        input_file = dataset['filename']
        name = dataset['dataset'] + '_' + dataset['exp']
        cube = iris.load_cube(input_file)
        start_year, end_year = dataset['start_year'], dataset['end_year']
        ##

        data = xr.DataArray.from_iris(cube)
        ds_model_ann = data.groupby('time.year').mean('time')
        ds_model_ann_trend_31 = calc_trend(ds_model_ann, start_year, end_year-31, 31)

        ds_model_ann_trend_31.to_netcdf(os.path.join(cfg["work_dir"], f"{dataset}_trend_31yr{start_year}_{end_year-31}.nc"))

if __name__ == "__main__":
    with run_diagnostic() as config:
        main(config)
