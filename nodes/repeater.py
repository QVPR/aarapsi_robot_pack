#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

def cmdvel_callback(msg):
    if msg.angular.x < 0.01:
        global msg_in, msg_time_old, msg_received
        msg_in = msg
        msg_in.angular.x = 0.05
        msg_time_old = msg_time
        msg_received = True


if __name__ == '__main__':
    try:
        rospy.init_node("repeater", anonymous=True)
        rospy.loginfo('Starting %s node.' % ("repeater"))

        rate_num = 80.0
        rate_obj = rospy.Rate(rate_num)

        msg_in = Twist
        msg_time = 0
        msg_time_old = 0
        msg_received = False

        topic_in = "/jackal_velocity_controller/cmd_vel"
        topic_out = "/jackal_velocity_controller/cmd_vel"

        sub = rospy.Subscriber(topic_in, Twist, cmdvel_callback, queue_size=1)
        pub = rospy.Publisher(topic_out, Twist, queue_size=1)

        while not rospy.is_shutdown():
            rate_obj.sleep()
            msg_time = rospy.Time.now().to_sec()
            if (not msg_received): continue
            if (msg_time - msg_time_old) > (1/rate_num):
                rospy.loginfo("Old message, do repeat")
                pub.publish(msg_in)
            rospy.loginfo("Diff: %0.2f" % (msg_time - msg_time_old))
            
        rospy.loginfo("Exit state reached.")
    except rospy.ROSInterruptException:
        pass
