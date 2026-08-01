import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import numpy as np
import time

class WallFollower(Node):
    DESIRED_DIST   = 1.0
    KP             = 1.2
    KD             = 0.09
    SPEED          = 9.2
    LOOKAHEAD_DIST = 1.0
    MAX_STEER      = 0.4
    LAP_THRESHOLD  = 1.0
    COLLISION_DIST = 0.15   # metres — if any LiDAR ray closer than this = collision

    def __init__(self):
        super().__init__('wall_follower')
        self.sub      = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.pub      = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_cb, 10)

        self.prev_error  = 0.0

        # Lap tracking
        self.lap_count   = 0
        self.lap_start   = None
        self.total_start = None
        self.near_start  = False
        self.start_x     = None
        self.start_y     = None
        self.position_x  = 0.0
        self.position_y  = 0.0

        # Collision tracking
        self.collision_count      = 0   # total collisions
        self.lap_collision_count  = 0   # collisions in current lap
        self.lap_collisions       = []  # collisions per lap history
        self.in_collision         = False  # debounce flag

        self.get_logger().info('Wall follower started — lap timer + collision counter ready!')

    def odom_cb(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        if self.start_x is None:
            self.start_x = self.position_x
            self.start_y = self.position_y
            self.get_logger().info(
                f'Start point set: ({self.start_x:.2f}, {self.start_y:.2f})')
        self.check_lap(self.position_x, self.position_y)

    def check_collision(self, scan):
        ranges = np.array(scan.ranges)
        ranges = ranges[np.isfinite(ranges)]
        if len(ranges) == 0:
            return
        min_dist = np.min(ranges)
        if min_dist < self.COLLISION_DIST:
            if not self.in_collision:
                self.in_collision        = True
                self.collision_count    += 1
                self.lap_collision_count += 1
                self.get_logger().info(
                    f'COLLISION! Lap {max(self.lap_count,1)} | '
                    f'Lap collisions: {self.lap_collision_count} | '
                    f'Total: {self.collision_count}')
        else:
            self.in_collision = False

    def check_lap(self, x, y):
        if self.start_x is None:
            return
        dist_to_start = ((x - self.start_x)**2 + (y - self.start_y)**2)**0.5
        if dist_to_start < self.LAP_THRESHOLD:
            if not self.near_start:
                self.near_start = True
                if self.lap_count == 0:
                    self.lap_start   = time.time()
                    self.total_start = time.time()
                    self.lap_count   = 1
                    self.get_logger().info(f'Car Speed: {self.SPEED} m/s')
                    self.get_logger().info('--- LAP 1 STARTED ---')
                else:
                    lap_time   = time.time() - self.lap_start
                    total_time = time.time() - self.total_start

                    # Save this lap's collision count
                    self.lap_collisions.append(self.lap_collision_count)
                    self.get_logger().info(
                        f'LAP {self.lap_count} COMPLETE | '
                        f'Lap: {lap_time:.2f}s | '
                        f'Total: {total_time:.2f}s | '
                        f'Collisions this lap: {self.lap_collision_count}')

                    # Reset lap collision counter
                    self.lap_collision_count = 0
                    self.lap_start = time.time()
                    self.lap_count += 1

                    if self.lap_count - 1 == 3:
                        total_time = time.time() - self.total_start
                        self.get_logger().info('─' * 50)
                        self.get_logger().info(f'3 LAPS DONE! Total: {total_time:.2f}s | Avg: {total_time/3:.2f}s')
                        self.get_logger().info(f'Collision Summary:')
                        for i, c in enumerate(self.lap_collisions):
                            self.get_logger().info(f'   Lap {i+1}: {c} collision(s)')
                        self.get_logger().info(f'   Total collisions: {self.collision_count}')
                        self.get_logger().info('─' * 50)
        else:
            self.near_start = False

    def get_range(self, scan, angle_deg):
        angle_rad = np.radians(angle_deg)
        idx = int((angle_rad - scan.angle_min) / scan.angle_increment)
        idx = np.clip(idx, 0, len(scan.ranges) - 1)
        r = scan.ranges[idx]
        if np.isinf(r) or np.isnan(r):
            return scan.range_max
        return r

    def get_wall_distance(self, scan):
        a = self.get_range(scan, -45)
        b = self.get_range(scan, -90)
        theta = np.radians(45)
        alpha = np.arctan2(a * np.cos(theta) - b, a * np.sin(theta))
        dist_now   = b * np.cos(alpha)
        dist_ahead = dist_now + self.LOOKAHEAD_DIST * np.sin(alpha)
        return dist_ahead

    def scan_cb(self, scan):
        # Check collision
        self.check_collision(scan)

        dist = self.get_wall_distance(scan)
        error = self.DESIRED_DIST - dist
        derivative = error - self.prev_error
        steering = self.KP * error + self.KD * derivative
        steering = np.clip(steering, -self.MAX_STEER, self.MAX_STEER)
        self.prev_error = error
        speed = self.SPEED * (1.0 - 0.5 * abs(steering) / self.MAX_STEER)

        msg = AckermannDriveStamped()
        msg.drive.speed          = speed
        msg.drive.steering_angle = steering
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
