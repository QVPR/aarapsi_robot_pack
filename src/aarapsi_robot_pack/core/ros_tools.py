#!/usr/bin/env python3
import rospy

class ROS_Param:

    updates_queued = []
    updates_possible = []

    def __init__(self, name, value, evaluation, force=False, namespace=None):
        if namespace is None:
            self.name = name
        else:
            self.name = namespace + "/" + name
        self.updates_possible.append(self.name)
        self.evaluation = evaluation
        self.value = None
        if rospy.has_param(self.name) and (not force):
            try:
                check_value = self.evaluation(rospy.get_param(self.name, value))
                self.value = check_value
            except:
                pass
        else:
            self.set(value)

    def get(self):
        if self.name in self.updates_queued:
            self.updates_queued.remove(self.name)
            try:
                check_value = self.evaluation(rospy.get_param(self.name, self.value))
                self.value = check_value
            except:
                pass
        return self.value

    def set(self, value):
        rospy.set_param(self.name, value)
        self.value = self.evaluation(value)