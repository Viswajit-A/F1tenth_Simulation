import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv, os, math

class WaypointRecorder(Node):
    SAVE_PATH = '/sim_ws/src/pure_pursuit/waypoints/waypoints.csv'
    MIN_DIST  = 0.1   # metres between saved waypoints

    def __init__(self):
        super().__init__('waypoint_recorder')
        os.makedirs(os.path.dirname(self.SAVE_PATH), exist_ok=True)
        self.file = open(self.SAVE_PATH, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['x', 'y', 'yaw'])
        self.last_x = self.last_y = None
        self.count = 0
        self.sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.get_logger().info(f'Recording waypoints → {self.SAVE_PATH}')

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))

        if self.last_x is None or \
           math.hypot(x - self.last_x, y - self.last_y) >= self.MIN_DIST:
            self.writer.writerow([f'{x:.4f}', f'{y:.4f}', f'{yaw:.4f}'])
            self.last_x, self.last_y = x, y
            self.count += 1
            if self.count % 50 == 0:
                self.get_logger().info(f'  {self.count} waypoints recorded...')

    def destroy_node(self):
        self.file.close()
        self.get_logger().info(
            f'Saved {self.count} waypoints to {self.SAVE_PATH}')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WaypointRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
