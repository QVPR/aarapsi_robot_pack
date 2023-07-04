#!/usr/bin/env python3

import rospy
import argparse as ap
import sys
from std_msgs.msg import String

from pyaarapsi.core.ros_tools       import NodeState, roslogger, LogType, SubscribeListener
from pyaarapsi.core.helper_tools    import formatException
from pyaarapsi.vpr_classes.base     import Base_ROS_Class, base_optional_args

'''
ROS Parameter Server Watcher

Our replacement for the mainstream dynamic parameter updates node/s.
This node observes changes to parameters in the /namespace, and then 
reports changes to each parameter on the /namespace/params_update 
topic. 

'''

def param_dict_key_extractor(_dict, namespace):
    def extract_dict(dict_in, namespace):
        _out = {}
        for k,v in zip(dict_in.keys(), dict_in.values()):
            if isinstance(v, dict):
                _out.update(extract_dict(v, namespace + '/' + k))
            else:
                _out[namespace + '/' + k] = v
        return _out
    dict_done = extract_dict(_dict, namespace)
    return dict_done

class Main_ROS_Class(Base_ROS_Class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, throttle=30)

        self.init_params(kwargs['rate_num'], kwargs['log_level'], kwargs['reset'])
        self.init_vars()
        self.init_rospy()

        self.node_ready(kwargs['order_id'])

    def init_vars(self):
        super().init_vars()
        self.watch_params   = param_dict_key_extractor(rospy.get_param(self.namespace), self.namespace)
        self.print("Watching params: %s" % str(self.watch_params.keys()), LogType.DEBUG)

    def init_rospy(self):
        super().init_rospy()
        self.watch_pub      = self.add_pub(self.namespace + "/params_update", String, queue_size=100)

    def watch(self):
        new_keys = []
        new_params_list     = param_dict_key_extractor(rospy.get_param(self.namespace), self.namespace)
        for key, value in zip(new_params_list.keys(), new_params_list.values()):
            if not key in self.watch_params.keys():
                new_keys.append(key)
                self.watch_params[key] = value
                self.watch_pub.publish(String(data=key))
            else:
                if not self.watch_params[key] == value:
                    self.print("Update detected for: %s (%s->%s)" % (key, str(self.watch_params[key]), str(value)), LogType.DEBUG)
                    self.watch_params[key] = value
                    self.watch_pub.publish(String(data=key))

        if len(new_keys):
            self.print('New params: %s' % str(new_keys))

    def main(self):
        self.set_state(NodeState.MAIN)

        while not rospy.is_shutdown():
            self.watch()
            self.rate_obj.sleep()

def do_args():
    parser = ap.ArgumentParser(prog="param_watcher.py", 
                            description="ROS Parameter Server Watcher",
                            epilog="Maintainer: Owen Claxton (claxtono@qut.edu.au)")
    
    # Optional Arguments:
    parser = base_optional_args(parser, node_name='param_watcher')
    
    # Parse args...
    return vars(parser.parse_known_args()[0])

if __name__ == '__main__':
    try:
        args = do_args()
        nmrc = Main_ROS_Class(**args)
        nmrc.print("Initialisation complete.", LogType.INFO)
        nmrc.main()
        nmrc.print("Operation complete.", LogType.INFO, ros=False) # False as rosnode likely terminated
        sys.exit()
    except SystemExit as e:
        pass
    except ConnectionRefusedError as e:
        roslogger("Error: Is the roscore running and accessible?", LogType.ERROR, ros=False) # False as rosnode likely terminated
    except:
        roslogger("Error state reached, system exit triggered.", LogType.WARN, ros=False) # False as rosnode likely terminated
        roslogger(formatException(), LogType.ERROR, ros=False)