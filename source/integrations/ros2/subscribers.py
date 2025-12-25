from geometry_msgs.msg import Twist
from rclpy.node import Node
import numpy as np


class TwistSubscriber(Node):
    """ROS2 subscriber for velocity commands."""
    def __init__(self, topic_name='cmd_vel'):
        super().__init__('twist_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            topic_name,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Store latest velocity commands
        self.velocity = 0.0
        self.angular = 0.0

    def listener_callback(self, msg):
        """Callback for velocity command messages."""
        self.velocity = msg.linear.x
        self.angular = msg.angular.z
        # self.get_logger().info(f'Received: linear={self.velocity:.2f}, angular={self.angular:.2f}')