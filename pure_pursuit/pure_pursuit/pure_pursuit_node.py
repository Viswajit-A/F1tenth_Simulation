import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import csv, math, numpy as np
import time

class PurePursuit(Node):
    LOOKAHEAD     = 2.0
    WHEELBASE     = 0.3302
    MAX_STEER     = 0.4
    SPEED_GAIN    = 1.0
    WAYPOINT_FILE = '/sim_ws/src/pure_pursuit/waypoints/waypoints_speed.csv'
    LAP_THRESHOLD = 1.0   # metres — how close to start point to count a lap

    def __init__(self):
        super().__init__('pure_pursuit')
        self.waypoints, self.speeds = self.load_waypoints()
        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints | speed {self.speeds.min():.1f}–{self.speeds.max():.1f} m/s')

        self.start_point = self.waypoints[0]   # first waypoint = start/finish line
        self.lap_count   = 0
        self.lap_start   = time.time()
        self.total_start = time.time()
        self.near_start  = False               # debounce flag

        self.sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        self.get_logger().info('Starting — lap timer ready!')

    def load_waypoints(self):
        pts, spds = [], []
        with open(self.WAYPOINT_FILE) as f:
            for row in csv.DictReader(f):
                pts.append([float(row['x']), float(row['y'])])
                spds.append(float(row['speed']))
        return np.array(pts), np.array(spds)

    def check_lap(self, x, y):
        dist_to_start = math.hypot(x - self.start_point[0], y - self.start_point[1])

        if dist_to_start < self.LAP_THRESHOLD:
            if not self.near_start:
                self.near_start = True
                if self.lap_count == 0:
                    # first crossing — just mark start
                    self.lap_start = time.time()
                    self.total_start = time.time()
                    self.lap_count = 1
                    self.get_logger().info('--- LAP 1 STARTED ---')
                else:
                    lap_time = time.time() - self.lap_start
                    total_time = time.time() - self.total_start
                    self.lap_count += 1
                    self.get_logger().info(
                        f' LAP {self.lap_count - 1} COMPLETE | '
                        f'Lap time: {lap_time:.2f}s | '
                        f'Total: {total_time:.2f}s')
                    self.lap_start = time.time()

                    if self.lap_count - 1 == 3:
                        self.get_logger().info(
                            f' 3 LAPS DONE! Total time: {total_time:.2f}s | '
                            f'Avg lap: {total_time/3:.2f}s')
        else:
            self.near_start = False   # reset debounce when away from start

    def odom_cb(self, msg):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        # Check lap
        self.check_lap(x, y)

        target_idx, target_pt = self.find_lookahead_point(x, y)
        if target_pt is None:
            return

        dx = target_pt[0] - x
        dy = target_pt[1] - y
        local_y  = -math.sin(yaw)*dx + math.cos(yaw)*dy
        curvature = 2.0 * local_y / (self.LOOKAHEAD ** 2)
        steering  = np.clip(math.atan(self.WHEELBASE * curvature), -self.MAX_STEER, self.MAX_STEER)
        speed = float(self.speeds[target_idx] * self.SPEED_GAIN)

        msg_out = AckermannDriveStamped()
        msg_out.drive.speed          = speed
        msg_out.drive.steering_angle = float(steering)
        self.pub.publish(msg_out)

    def find_lookahead_point(self, x, y):
        dists = np.hypot(self.waypoints[:, 0] - x, self.waypoints[:, 1] - y)
        closest = np.argmin(dists)
        n = len(self.waypoints)
        for i in range(n):
            idx = (closest + i) % n
            if dists[idx] >= self.LOOKAHEAD:
                return idx, self.waypoints[idx]
        fallback = (closest + 10) % n
        return fallback, self.waypoints[fallback]

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
