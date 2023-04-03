import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from bokeh.plotting import figure
from bokeh.palettes import Sunset11, Viridis256
from bokeh.models import ColumnDataSource, Range1d
from scipy.spatial.distance import cdist
import warnings
import cv2

#   _____       _____  _       _   
#  |  __ \     |  __ \| |     | |  
#  | |__) |   _| |__) | | ___ | |_ 
#  |  ___/ | | |  ___/| |/ _ \| __|
#  | |   | |_| | |    | | (_) | |_ 
#  |_|    \__, |_|    |_|\___/ \__|
#          __/ |                   
#         |___/                    

##################################################################
#### Sliding Similarity Matrix Figure: do and update

def doMtrxFig(axes, odom_in):
    plt.sca(axes)
    mtrx_image = np.zeros((len(odom_in['position']['x']), len(odom_in['position']['x'])))
    mtrx_handle = axes.imshow(mtrx_image)
    axes.set(xlabel='Query Frame', ylabel='Reference Frame')

    return {'img': mtrx_image, 'handle': mtrx_handle}

def updateMtrxFig(mInd, tInd, dvc, odom_in, fig_handles):
    img_new = np.delete(fig_handles['img'], 0, 1) # delete first column (oldest query)
    fig_handles['img'] = np.concatenate((img_new, np.array(dvc)), 1)
    fig_handles['handle'].set_data(fig_handles['img']) # TODO: access directly.
    fig_handles['handle'].autoscale() # https://stackoverflow.com/questions/10970492/matplotlib-no-effect-of-set-data-in-imshow-for-the-plot

##################################################################
#### Distance Vector Figure: do and update

def doDVecFig(axes, odom_in):
# Set up distance vector figure
# https://www.geeksforgeeks.org/data-visualization-using-matplotlib/
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

    plt.sca(axes) # distance vector
    dist_vector = plt.plot([], [], 'k-')[0] # distance vector
    lowest_dist = plt.plot([], [], 'ro', markersize=7)[0] # matched image (lowest distance)
    actual_dist = plt.plot([], [], 'mo', markersize=7)[0] # true image (correct match)
    axes.set(xlabel='Index', ylabel='Distance')
    axes.legend(["Image Distances", "Selected", "True"])
    axes.set_xlim(0, len(odom_in['position']['x']))
    axes.set_ylim(0, 1.2)

    return {'axes': axes, 'dis': dist_vector, 'low': lowest_dist, 'act': actual_dist}

def updateDVecFig(mInd, tInd, dvc, odom_in, fig_handles):
# Update DVec figure with new data (match->mInd, true->tInd)
# update (overwrite) visualisation with new data:

    # overwrite with new distance vector / image distance:
    max_val = np.max(dvc[:])
    fig_handles['dis'].set_xdata(range(len(dvc)))
    fig_handles['dis'].set_ydata(dvc/max_val)
    # overwrite with new lowest match:
    fig_handles['low'].set_xdata(mInd)
    fig_handles['low'].set_ydata(dvc[mInd]/max_val)
    # overwrite with new truth value:
    fig_handles['act'].set_xdata(tInd)
    fig_handles['act'].set_ydata(dvc[tInd]/max_val)

##################################################################
#### Odometry Figure: do and update

def doOdomFig(axes, odom_in):
# Set up odometry figure

    plt.sca(axes)
    ref_plotted = plt.plot(odom_in['position']['x'], odom_in['position']['y'], 'b-')[0]
    mat_plotted = plt.plot([], [], 'r+', markersize=6)[0] # Match values: init as empty
    tru_plotted = plt.plot([], [], 'gx', markersize=4)[0] # True value: init as empty

    axes.set(xlabel='X-Axis', ylabel='Y-Axis')
    axes.legend(["Reference", "Match", "True"])
    axes.set_aspect('equal')

    return {'axes': axes, 'ref': ref_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateOdomFig(mInd, tInd, dvc, odom_in, fig_handles):
# Update odometryfigure with new data (match->mInd, true->tInd)
# Use old handles (reference, match, true)
    # Only display last 'queries_keep' number of points
    num_queries = len(list(fig_handles['tru'].get_xdata()))
    queries_keep = 10
    start_ind = num_queries - queries_keep + 1
    if num_queries < queries_keep:
        start_ind = 0
        
    ## odometry plot:
    # Append new value for "match" (what it matched the image to)
    fig_handles['mat'].set_xdata(np.append(fig_handles['mat'].get_xdata()[start_ind:num_queries], odom_in['position']['x'][mInd]))
    fig_handles['mat'].set_ydata(np.append(fig_handles['mat'].get_ydata()[start_ind:num_queries], odom_in['position']['y'][mInd]))
    # Append new value for "true" (what it should be from the robot odom)
    fig_handles['tru'].set_xdata(np.append(fig_handles['tru'].get_xdata()[start_ind:num_queries], odom_in['position']['x'][tInd]))
    fig_handles['tru'].set_ydata(np.append(fig_handles['tru'].get_ydata()[start_ind:num_queries], odom_in['position']['y'][tInd]))

#   ____        _        _     
#  |  _ \      | |      | |    
#  | |_) | ___ | | _____| |__  
#  |  _ < / _ \| |/ / _ \ '_ \ 
#  | |_) | (_) |   <  __/ | | |
#  |____/ \___/|_|\_\___|_| |_|
    
def disable_toolbar(fig, interact=False):
    # hide toolbar and disable plot interaction
    fig.toolbar_location = None
    if not interact:
        fig.toolbar.active_drag = None
        fig.toolbar.active_scroll = None
        fig.toolbar.active_tap = None
    return fig

def get_contour_data(X, Y, Z, levels):
    cs = plt.contour(X, Y, Z, levels)
    xs = []
    ys = []
    xt = []
    yt = []
    col = []
    text = []
    isolevelid = 0
    for isolevel in cs.collections:
        isocol = isolevel.get_color()[0]
        thecol = 3 * [None]
        theiso = str(cs.get_array()[isolevelid])
        isolevelid += 1
        for i in range(3):
            thecol[i] = int(255 * isocol[i])
        thecol = '#%02x%02x%02x' % (thecol[0], thecol[1], thecol[2])

        for path in isolevel.get_paths():
            v = path.vertices
            x = v[:, 0]
            y = v[:, 1]
            xs.append(x.tolist())
            ys.append(y.tolist())
            try:
                xt.append(x[len(x) / 2])
                yt.append(y[len(y) / 2])
            except:
                xt.append(5)
                yt.append(5)
            text.append(theiso)
            col.append(thecol)

    source = ColumnDataSource(data={'xs': xs, 'ys': ys, 'line_color': col,'xt':xt,'yt':yt,'text':text})
    return source

##################################################################
#### Contour Figure: do and update

def doCntrFigBokeh(nmrc, odom_in):
# Set up contour figure

    fig_cntr        = figure(title="SVM Contour", width=500, height=500, \
                            x_axis_label = 'VA Factor', y_axis_label = 'Grad Factor')

    fig_cntr        = disable_toolbar(fig_cntr)

    img_rand        = np.array(np.ones((1000,1000,4))*255, dtype=np.uint8)
    img_uint32      = img_rand.view(dtype=np.uint32).reshape(img_rand.shape[:-1])
    img_ds          = ColumnDataSource(data=dict(image=[img_uint32], x=[0], y=[0], dw=[10], dh=[10])) #CDS must contain columns, hence []
    img_plotted     = fig_cntr.image_rgba(image='image', x='x', y='y', dw='dw', dh='dh', source=img_ds)
    in_yes_plotted  = fig_cntr.circle(x=[], y=[], color="green",  size=8, alpha=0.4, legend_label="True Positive")
    out_yes_plotted = fig_cntr.circle(x=[], y=[], color="red",    size=8, alpha=0.4, legend_label="True Negative")
    in_no_plotted   = fig_cntr.circle(x=[], y=[], color="blue",   size=8, alpha=0.4, legend_label="False Positive")
    out_no_plotted  = fig_cntr.circle(x=[], y=[], color="orange", size=8, alpha=0.4, legend_label="False Negative")

    fig_cntr.x_range.range_padding = 0
    fig_cntr.y_range.range_padding = 0

    return {'fig': fig_cntr, 'img': img_plotted, 'in_y': in_yes_plotted, 'out_y': out_yes_plotted, 'in_n': in_no_plotted, 'out_n': out_no_plotted}

def updateCntrFigBokeh(nmrc, mInd, tInd, dvc, odom_in):

    xlims = (np.min(nmrc.svm_field_msg.data.xlim), np.max(nmrc.svm_field_msg.data.xlim))
    ylims = (np.min(nmrc.svm_field_msg.data.ylim), np.max(nmrc.svm_field_msg.data.ylim))

    f1_clip = np.clip(nmrc.state.factors[0], xlims[0], xlims[1])
    f2_clip = np.clip(nmrc.state.factors[1], ylims[0], ylims[1])

    to_stream = dict(x=[f1_clip], y=[f2_clip])

    data_to_add = ''
    if nmrc.state.mStateBin:
        data_to_add = 'in'
    else:
        data_to_add = 'out'
    if nmrc.state.data.state == 2:
        data_to_add += '_y'
    elif nmrc.state.data.state == 1:
        data_to_add += '_n'

    nmrc.fig_cntr_handles[data_to_add].data_source.stream(to_stream, rollover = 50)

    if not nmrc.new_field:
        return
    
    ros_msg_img = nmrc.svm_field_msg.image
    if nmrc.COMPRESS_IN.get():
        cv_msg_img = nmrc.bridge.compressed_imgmsg_to_cv2(ros_msg_img, "passthrough")
    else:
        cv_msg_img = nmrc.bridge.imgmsg_to_cv2(ros_msg_img, "passthrough")

    # process image from three layer (rgb) into four layer (rgba) uint8:
    img_rgba = np.array(np.dstack((np.flipud(np.flip(cv_msg_img,2)), np.ones((1000,1000))*255)), dtype=np.uint8)
    # collapse into uint32:
    img_uint32 = img_rgba.view(dtype=np.uint32).reshape(img_rgba.shape[:-1])

    nmrc.fig_cntr_handles['img'].data_source.data = dict(x=[xlims[0]], y=[ylims[0]], dw=[xlims[1] - xlims[0]], \
                                                         dh=[ylims[1] - ylims[0]], image=[img_uint32.copy()])

    nmrc.fig_cntr_handles['fig'].title.text         = nmrc.svm_field_msg.data.title
    nmrc.fig_cntr_handles['fig'].xaxis.axis_label   = nmrc.svm_field_msg.data.xlab
    nmrc.fig_cntr_handles['fig'].yaxis.axis_label   = nmrc.svm_field_msg.data.ylab
    nmrc.fig_cntr_handles['fig'].x_range            = Range1d(xlims[0], xlims[1])
    nmrc.fig_cntr_handles['fig'].y_range            = Range1d(ylims[0], ylims[1])

    nmrc.new_field = False

##################################################################
#### Distance Vector Figure: do and update

def doDVecFigBokeh(nmrc, odom_in):
# Set up distance vector figure

    fig_dvec    = figure(title="Distance Vector", width=500, height=250, \
                            x_axis_label = 'Index', y_axis_label = 'Distance', \
                            x_range = (0, len(odom_in['position']['x'])), y_range = (0, 1.2))
    fig_dvec    = disable_toolbar(fig_dvec)
    spd_plotted = fig_dvec.line([],   [], color="orange",           legend_label="Spatial Separation") # Distance from match
    dvc_plotted = fig_dvec.line([],   [], color="black",            legend_label="Distance Vector") # distance vector
    mat_plotted = fig_dvec.circle([], [], color="red",     size=7,  legend_label="Selected") # matched image (lowest distance)
    tru_plotted = fig_dvec.circle([], [], color="magenta", size=7,  legend_label="True") # true image (correct match)

    fig_dvec.legend.location=(0, 140)
    fig_dvec.legend.orientation='horizontal'
    fig_dvec.legend.border_line_alpha=0
    fig_dvec.legend.background_fill_alpha=0

    return {'fig': fig_dvec, 'spd': spd_plotted, 'dvc': dvc_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateDVecFigBokeh(nmrc, mInd, tInd, dvc, odom_in):
# Update DVec figure with new data (match->mInd, true->tInd)
# Use old handles (mat, tru) and crunched distance vector (dvc)
    spd = cdist(np.transpose(np.matrix([odom_in['position']['x'],odom_in['position']['y']])), \
        np.matrix([odom_in['position']['x'][mInd], odom_in['position']['y'][mInd]]))
    spd_max_val = np.max(spd[:])
    dvc_max_val = np.max(dvc[:])
    nmrc.fig_dvec_handles['spd'].data_source.data = {'x': list(range(len(spd-1))), 'y': spd/spd_max_val}
    nmrc.fig_dvec_handles['dvc'].data_source.data = {'x': list(range(len(dvc-1))), 'y': dvc/dvc_max_val}
    nmrc.fig_dvec_handles['mat'].data_source.data = {'x': [mInd], 'y': [dvc[mInd]/dvc_max_val]}
    nmrc.fig_dvec_handles['tru'].data_source.data = {'x': [tInd], 'y': [dvc[tInd]/dvc_max_val]}

##################################################################
#### Distance Vector Figure: do and update

def doFDVCFigBokeh(nmrc, odom_in):
# Set up distance vector figure

    fig_dvec    = figure(title="Distance Vector", width=500, height=250, \
                            x_axis_label = 'Index', y_axis_label = 'Distance', \
                            x_range = (0, len(odom_in['position']['x'])), y_range = (0, 1.2))
    fig_dvec    = disable_toolbar(fig_dvec)
    dvc_plotted = fig_dvec.line([],   [], color="black",            legend_label="Warped Distance Vector") # distance vector
    mat_plotted = fig_dvec.circle([], [], color="red",     size=7,  legend_label="Selected") # matched image (lowest distance)
    tru_plotted = fig_dvec.circle([], [], color="magenta", size=7,  legend_label="True") # true image (correct match)

    fig_dvec.legend.location=(0, 140)
    fig_dvec.legend.orientation='horizontal'
    fig_dvec.legend.border_line_alpha=0
    fig_dvec.legend.background_fill_alpha=0

    return {'fig': fig_dvec, 'dvc': dvc_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateFDVCFigBokeh(nmrc, mInd, tInd, dvc, odom_in):
# Update DVec figure with new data (match->mInd, true->tInd)
# Use old handles (mat, tru) and crunched distance vector (dvc)
    spd = cdist(np.transpose(np.matrix([odom_in['position']['x'],odom_in['position']['y']])), \
        np.matrix([odom_in['position']['x'][mInd], odom_in['position']['y'][mInd]]))
    spd_max_val = np.max(spd[:])
    dvc_max_val = np.max(dvc[:])
    spd_norm = np.array(spd).flatten()/spd_max_val 
    dvc_norm = np.array(dvc).flatten()/dvc_max_val
    spd_x_dvc = (spd_norm**2 + dvc_norm) / 2
    nmrc.fig_fdvc_handles['dvc'].data_source.data = {'x': list(range(len(dvc-1))), 'y': spd_x_dvc}
    nmrc.fig_fdvc_handles['mat'].data_source.data = {'x': [mInd], 'y': [spd_x_dvc[mInd]]}
    nmrc.fig_fdvc_handles['tru'].data_source.data = {'x': [tInd], 'y': [spd_x_dvc[tInd]]}


##################################################################
#### Odometry Figure: do and update

def doOdomFigBokeh(nmrc, odom_in):
# Set up odometry figure

    xlims = (np.min(odom_in['position']['x']), np.max(odom_in['position']['x']))
    ylims = (np.min(odom_in['position']['y']), np.max(odom_in['position']['y'] * 1.1))
    xrang = xlims[1] - xlims[0]
    yrang = ylims[1] - ylims[0]

    fig_odom    = figure(title="Odometries", width=500, height=250, \
                            x_axis_label = 'X-Axis', y_axis_label = 'Y-Axis', \
                            x_range = (xlims[0] - 0.1 * xrang, xlims[1] + 0.1 * xrang), \
                            y_range = (ylims[0] - 0.1 * yrang, ylims[1] + 0.1 * yrang), \
                            match_aspect = True, aspect_ratio = "auto")
    fig_odom    = disable_toolbar(fig_odom)

    # Make legend glyphs
    fig_odom.line(x=[xlims[1]*2], y=[ylims[1]*2], color="blue", line_dash='dotted', legend_label="Path")
    fig_odom.cross(x=[xlims[1]*2], y=[ylims[1]*2], color="red", legend_label="Match", size=14)
    fig_odom.circle( x=[xlims[1]*2], y=[ylims[1]*2], color="green", legend_label="True", size=4,)
    
    ref_plotted = fig_odom.line(   x=odom_in['position']['x'], y=odom_in['position']['y'], color="blue", \
                                   alpha=0.5, line_dash='dotted')
    var_plotted = fig_odom.circle( x=[], y=[], color="blue", size=[], alpha=0.1)
    seg_plotted = fig_odom.segment(x0=[], y0=[], x1=[], y1=[], line_color="black", line_width=1, alpha=[])
    mat_plotted = fig_odom.cross(  x=[], y=[], color="red", size=12, alpha=[])
    tru_plotted = fig_odom.circle( x=[], y=[], color="green", size=4, alpha=1.0)

    fig_odom.legend.location= (120, 145)
    fig_odom.legend.orientation='horizontal'
    fig_odom.legend.border_line_alpha=0
    fig_odom.legend.background_fill_alpha=0
    fig_odom.legend.items[0].visible = True
    fig_odom.legend.items[1].visible = True
    fig_odom.legend.items[2].visible = True

    return {'fig': fig_odom, 'ref': ref_plotted, 'var': var_plotted, 'seg': seg_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateOdomFigBokeh(nmrc, mInd, tInd, dvc, odom_in):
# Update odometryfigure with new data (match->mInd, true->tInd)
# Use old handles (reference, match, true)
    # Stream/append new value for "match" (estimate) and "true" (correct) odometry:
    num_points  = len(odom_in['position']['y'])
    separation  = float(np.min([abs(mInd-tInd), (-abs(mInd-tInd)%num_points)],0) / (num_points / 2))

    new_alpha   = np.round(0.9 * separation,3)
    new_size    = np.round(4.0 * np.sqrt(20.0 * separation),3)

    new_tru_data = dict(x=[np.round(odom_in['position']['x'][tInd],3)], y=[np.round(odom_in['position']['y'][tInd],3)])
    new_var_data = dict(**new_tru_data, size=[new_size])

    new_mat_data = dict(x=[np.round(odom_in['position']['x'][mInd],3)], y=[np.round(odom_in['position']['y'][mInd],3)], \
                        fill_alpha=[new_alpha], hatch_alpha=[new_alpha], line_alpha=[new_alpha])
    new_mod_data = dict(x0=new_mat_data['x'], y0=new_mat_data['y'], x1=new_tru_data['x'], y1=new_tru_data['y'], \
                        line_alpha=[0.05])
    
    with warnings.catch_warnings():
        # Bokeh gets upset because we have discrete data that we are streaming that is duplicate
        warnings.simplefilter("ignore")
        
        nmrc.fig_odom_handles['tru'].data_source.stream(new_tru_data, rollover=1)
        nmrc.fig_odom_handles['var'].data_source.stream(new_var_data, rollover=2*num_points)

        nmrc.fig_odom_handles['seg'].data_source.stream(new_mod_data, rollover=num_points)
        nmrc.fig_odom_handles['mat'].data_source.stream(new_mat_data, rollover=num_points)

##################################################################
#### Distance Vector Figure: do and update

def doSVMMFigBokeh(nmrc, odom_in):
# Set up distance vector figure

    fig_dvec    = figure(title="Distance Vector", width=500, height=250, \
                            x_axis_label = 'Index', y_axis_label = 'Distance', \
                            x_range = (0, len(odom_in['position']['x'])), y_range = (0, 1.2))
    fig_dvec    = disable_toolbar(fig_dvec)
    spd_plotted = fig_dvec.line([],   [], color="orange",           legend_label="Spatial Separation") # Distance from match
    dvc_plotted = fig_dvec.line([],   [], color="black",            legend_label="Distance Vector") # distance vector
    mat_plotted = fig_dvec.circle([], [], color="red",     size=7,  legend_label="Selected") # matched image (lowest distance)
    tru_plotted = fig_dvec.circle([], [], color="magenta", size=7,  legend_label="True") # true image (correct match)

    fig_dvec.legend.location=(0, 140)
    fig_dvec.legend.orientation='horizontal'
    fig_dvec.legend.border_line_alpha=0
    fig_dvec.legend.background_fill_alpha=0

    return {'fig': fig_dvec, 'spd': spd_plotted, 'dvc': dvc_plotted, 'mat': mat_plotted, 'tru': tru_plotted}

def updateSVMMFigBokeh(nmrc, mInd, tInd, dvc, odom_in):
# Update DVec figure with new data (match->mInd, true->tInd)
# Use old handles (mat, tru) and crunched distance vector (dvc)
    spd = cdist(np.transpose(np.matrix([odom_in['position']['x'],odom_in['position']['y']])), \
        np.matrix([odom_in['position']['x'][mInd], odom_in['position']['y'][mInd]]))
    spd_max_val = np.max(spd[:])
    dvc_max_val = np.max(dvc[:])
    nmrc.fig_svmm_handles['spd'].data_source.data = {'x': list(range(len(spd-1))), 'y': spd/spd_max_val}
    nmrc.fig_svmm_handles['dvc'].data_source.data = {'x': list(range(len(dvc-1))), 'y': dvc/dvc_max_val}
    nmrc.fig_svmm_handles['mat'].data_source.data = {'x': [mInd], 'y': [dvc[mInd]/dvc_max_val]}
    nmrc.fig_svmm_handles['tru'].data_source.data = {'x': [tInd], 'y': [dvc[tInd]/dvc_max_val]}
