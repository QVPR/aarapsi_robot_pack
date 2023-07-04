#!/usr/bin/env python3

import os

from random import random
from functools import partial
import time

from bokeh.layouts import column, row
from bokeh.models.widgets import Div
from bokeh.models import Button, ColumnDataSource, AjaxDataSource
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.themes import Theme

from pyaarapsi.core.ajax_tools          import GET_Method_Types, AJAX_Connection
from pyaarapsi.core.enum_tools          import enum_name
from pyaarapsi.core.helper_tools        import vis_dict
from pyaarapsi.vpr_simple.vpr_plots_new import doOdomFigBokeh, updateOdomFigBokeh

print("\n...........................\n")

ajax = AJAX_Connection(name='Bokeh')

while not ajax.check_if_ready():
    print('Waiting for AJAX database to finish initialisation...')
    time.sleep(1)

print("AJAX responsive.")

headers = {
            'Publisher-Name': 'bokeh', 
            'Method-Type': 'Get',
            'Data-Key': 'odom',
            }
source = AjaxDataSource(data_url='http://localhost:5050/data',
                        polling_interval=100, http_headers=headers, method="GET")

p = figure(height=300, width=800, background_fill_color="lightgrey",
           title="Streaming Noisy sin(x) via Ajax")
p.circle('px', 'py', source=source)
p.x_range.follow = "end"
p.x_range.follow_interval = 10

def test():
    print(source.data.keys())

#odom_fig = doOdomFigBokeh(source)
doc = curdoc()
doc.theme = Theme(filename=os.path.dirname(__file__)+"/theme.yaml")
doc.add_periodic_callback(test, 1)
doc.add_root(p)