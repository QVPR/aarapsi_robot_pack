#!/usr/bin/env python3

import os

from random import random
from functools import partial
import time

from bokeh.layouts import column, row
from bokeh.models.widgets import Div
from bokeh.models import Button, ColumnDataSource
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.themes import Theme

# def _callback(_in, doc):
#     print(time.strftime("%H:%M:%S",time.localtime()))

# def main(doc):
#     p = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
#     data = {'x_values': [1, 2, 3, 4, 5],
#         'y_values': [6, 7, 2, 3, 6]}

#     source = ColumnDataSource(data=data)
#     p.circle('x_values', 'y_values', source=source)
#     doc.add_root(column(p))
#     doc.theme = Theme(filename=os.path.dirname(os.path.abspath(__file__)) + "/theme.yaml")
#     doc.add_periodic_callback(partial(_callback, _in=1, doc=doc), 100)

# main(curdoc())

# Bokeh related code
from bokeh.models import AjaxDataSource

source = AjaxDataSource(data_url='http://localhost:5050/data', polling_interval=100, method="GET", http_headers={'Request-Type': 'bokeh/odom'})

p = figure(height=300, width=800, background_fill_color="lightgrey",
           title="Streaming Noisy sin(x) via Ajax")
p.circle('px', 'py', source=source)

#p.x_range.follow = "end"
#p.x_range.follow_interval = 10

curdoc().add_root(column(p))