import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import csv, math, time
import numpy as np

class StanleyController(Node):

    # --- Vehicle params ---
    WHEELBASE  = 0.3302
    MAX_STEER  = 0.4

    # --- Stanley params ---
    K          = 0.5    # cross-track error gain — increase = stronger correction
    K_SOFT     = 1.0    # softening constant — prevents division by zero at low speed
    SPEED_GAIN = 1.0    # scale all waypoint speeds

    # --- Path tracking ---
    LOOKAHEAD_IDX = 5   # waypoints ahead of closest for front axle reference

    # --- Lap tracking ---
    LAP_THRESHOLD  = 1.0
    COLLISION_DIST = 0.25

    WAYPOINT_FILE = '/sim_ws/src/pure_pursuit/waypoints/waypoints_speed.csv'

    def __init__(self):
        super().__init__('stanley_controller')

        self.waypoints, self.speeds = self.load_waypoints()
        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints | '
            f'speed {self.speeds.min():.1f}-{self.speeds.max():.1f} m/s')

        # State
        self.position_x    = 0.0
        self.position_y    = 0.0
        self.yaw           = 0.0
        self.current_speed = 0.0

        # Lap tracking
        self.start_x             = None
        self.start_y             = None
        self.lap_count           = 0
        self.lap_start           = None
        self.total_start         = None
        self.near_start          = False
        self.collision_count     = 0
        self.lap_collision_count = 0
        self.lap_collisions      = []
        self.in_collision        = False

        self.sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        self.get_logger().info('Stanley controller started!')

    # ─────────────────────────────────────────
    # Load waypoints
    # ─────────────────────────────────────────
    def load_waypoints(self):
        pts, spds = [], []
        with open(self.WAYPOINT_FILE) as f:
            for row in csv.DictReader(f):
                pts.append([float(row['x']), float(row['y'])])
                spds.append(float(row['speed']))
        return np.array(pts), np.array(spds)

    # ─────────────────────────────────────────
    # Find closest waypoint index
    # ─────────────────────────────────────────
    def find_closest(self, fx, fy):
        dists = np.hypot(
            self.waypoints[:, 0] - fx,
            self.waypoints[:, 1] - fy)
        return int(np.argmin(dists))

    # ─────────────────────────────────────────
    # Stanley control law
    # ─────────────────────────────────────────
    def stanley_control(self, x, y, yaw, speed):
        n = len(self.waypoints)

        # Front axle position
        fx = x + self.WHEELBASE * math.cos(yaw)
        fy = y + self.WHEELBASE * math.sin(yaw)

        # Find closest waypoint to front axle
        closest_idx = self.find_closest(fx, fy)
        ref_idx     = (closest_idx + self.LOOKAHEAD_IDX) % n

        # Reference point and path direction
        ref_x, ref_y = self.waypoints[ref_idx]
        next_idx     = (ref_idx + 1) % n
        path_dx      = self.waypoints[next_idx, 0] - ref_x
        path_dy      = self.waypoints[next_idx, 1] - ref_y
        path_yaw     = math.atan2(path_dy, path_dx)

        # 1. Heading error — difference between car yaw and path yaw
        heading_error = path_yaw - yaw
        # Normalize to [-pi, pi]
        heading_error = math.atan2(
            math.sin(heading_error),
            math.cos(heading_error))

        # 2. Cross-track error — signed perpendicular distance
        #    from front axle to the nearest path point
        dx  = fx - ref_x
        dy  = fy - ref_y
        # Project onto path normal
        cte = math.sin(path_yaw) * dx - math.cos(path_yaw) * dy

        # 3. Stanley steering law
        #    steering = heading_error + arctan(k * cte / (speed + k_soft))
        cte_correction = math.atan2(
            self.K * cte,
            self.K_SOFT + speed)

        steering = heading_error + cte_correction
        steering = np.clip(steering, -self.MAX_STEER, self.MAX_STEER)

        # Speed from waypoint profile
        speed_cmd = float(self.speeds[ref_idx] * self.SPEED_GAIN)

        return steering, speed_cmd, ref_idx, cte, heading_error

    # ─────────────────────────────────────────
    # Odometry callback — main control loop
    # ─────────────────────────────────────────
    def odom_cb(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        q               = msg.pose.pose.orientation
        self.yaw        = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

        # Set start point
        if self.start_x is None:
            self.start_x = self.position_x
            self.start_y = self.position_y
            self.get_logger().info(
                f'Start point: ({self.start_x:.2f}, {self.start_y:.2f})')

        # Lap check
        self.check_lap(self.position_x, self.position_y)

        # Stanley control
        steer, speed, ref_idx, cte, he = self.stanley_control(
            self.position_x,
            self.position_y,
            self.yaw,
            max(self.current_speed, 0.1))

        # Publish
        msg_out = AckermannDriveStamped()
        msg_out.drive.speed          = speed
        msg_out.drive.steering_angle = steer
        self.pub.publish(msg_out)

    # ─────────────────────────────────────────
    # Lap tracking
    # ─────────────────────────────────────────
    def check_lap(self, x, y):
        if self.start_x is None:
            return
        dist = math.hypot(x - self.start_x, y - self.start_y)

        if dist < self.LAP_THRESHOLD:
            if not self.near_start:
                self.near_start = True
                if self.lap_count == 0:
                    self.lap_start   = time.time()
                    self.total_start = time.time()
                    self.lap_count   = 1
                    self.get_logger().info(
                        f'Car Speed: {self.speeds.max():.1f} m/s')
                    self.get_logger().info('--- LAP 1 STARTED ---')
                else:
                    lap_time   = time.time() - self.lap_start
                    total_time = time.time() - self.total_start
                    self.lap_collisions.append(self.lap_collision_count)
                    self.lap_collision_count = 0
                    self.lap_count += 1
                    self.get_logger().info(
                        f'LAP {self.lap_count-1} COMPLETE | '
                        f'Lap: {lap_time:.2f}s | '
                        f'Total: {total_time:.2f}s | '
                        f'CTE avg: -- | '
                        f'Collisions: {self.lap_collisions[-1]}')
                    self.lap_start = time.time()

                    if self.lap_count - 1 == 3:
                        total_time = time.time() - self.total_start
                        self.get_logger().info('─' * 50)
                        self.get_logger().info(
                            f'3 LAPS DONE! Total: {total_time:.2f}s | '
                            f'Avg: {total_time/3:.2f}s')
                        self.get_logger().info('Collision summary:')
                        for i, c in enumerate(self.lap_collisions):
                            self.get_logger().info(
                                f'  Lap {i+1}: {c} collision(s)')
                        self.get_logger().info(
                            f'  Total: {self.collision_count}')
                        self.get_logger().info('─' * 50)
        else:
            self.near_start = False

def main(args=None):
    rclpy.init(args=args)
    node = StanleyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
