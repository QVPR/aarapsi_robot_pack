#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

params = rospy.get_param_names()

print(params)

params_vpr_cruncher = [i for i in params if i.startswith('/vpr_cruncher')]
params_vpr_monitor  = [i for i in params if i.startswith('/vpr_monitor')]
params_vpr_plotter  = [i for i in params if i.startswith('/vpr_plotter')]
params_vpr_nodes    = [i for i in params if i.startswith('/vpr_nodes')]

rospy.init_node("param_watcher", log_level=rospy.DEBUG, anonymous=True)

pub = rospy.Publisher("/vpr_nodes/params_update", String, queue_size=100)

rospy.logdebug("Cruncher params:")
rospy.logdebug(params_vpr_cruncher)
rospy.logdebug("Monitor params:")
rospy.logdebug(params_vpr_monitor)
rospy.logdebug("Plotter params:")
rospy.logdebug(params_vpr_plotter)
rospy.logdebug("Namespace params:")
rospy.logdebug(params_vpr_nodes)

watch_params = params_vpr_cruncher + params_vpr_monitor + params_vpr_plotter + params_vpr_nodes
params_dict = dict.fromkeys(watch_params)

rate_obj = rospy.Rate(1)

def watch(params_dict):
    for i in list(params_dict.keys()):
        check_param = rospy.get_param(i)
        if not params_dict[i] == check_param:
            rospy.loginfo("Update detected for: %s (%s->%s)" % (i, params_dict[i], check_param))
            params_dict[i] = check_param
            pub.publish(String(i))
    return params_dict

while not rospy.is_shutdown():
    params_dict.update(watch(params_dict))
    rate_obj.sleep()
