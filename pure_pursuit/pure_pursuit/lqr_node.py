import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import csv, math, time
import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController(Node):
    # --- Tuning ---
    WHEELBASE     = 0.3302
    MAX_STEER     = 0.4
    SPEED_GAIN    = 1.0
    LOOKAHEAD_IDX = 10      # waypoints ahead to use as reference
    LAP_THRESHOLD = 1.0

    # LQR cost matrices
    # Q: penalizes state error [cross-track, heading]
    # R: penalizes control effort [steering]
    Q = np.diag([10.0, 1.0])   # increase Q[0] = tighter path tracking
    R = np.array([[1.0]])       # increase R   = smoother steering

    WAYPOINT_FILE = '/sim_ws/src/pure_pursuit/waypoints/waypoints_speed.csv'

    def __init__(self):
        super().__init__('lqr_controller')
        self.waypoints, self.speeds = self.load_waypoints()
        self.K = self.compute_lqr_gain()
        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints | '
            f'speed {self.speeds.min():.1f}–{self.speeds.max():.1f} m/s | '
            f'LQR gain K={self.K}')

        # Lap tracking
        self.start_point = self.waypoints[0]
        self.lap_count   = 0
        self.lap_start   = time.time()
        self.total_start = time.time()
        self.near_start  = False

        self.sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        self.get_logger().info('LQR controller started!')

    def load_waypoints(self):
        pts, spds = [], []
        with open(self.WAYPOINT_FILE) as f:
            for row in csv.DictReader(f):
                pts.append([float(row['x']), float(row['y'])])
                spds.append(float(row['speed']))
        return np.array(pts), np.array(spds)

    def compute_lqr_gain(self):
        """
        Solve continuous-time LQR for bicycle model.
        State: [cross_track_error, heading_error]
        Input: [steering_angle]
        A and B matrices from linearized bicycle kinematics.
        """
        v = 3.0   # nominal speed for linearization
        A = np.array([[0, v],
                      [0, 0]])
        B = np.array([[0],
                      [v / self.WHEELBASE]])
        # Solve Riccati equation: A'P + PA - PBR⁻¹B'P + Q = 0
        P = solve_continuous_are(A, B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ B.T @ P
        return K

    def find_closest_waypoint(self, x, y):
        dists = np.hypot(
            self.waypoints[:, 0] - x,
            self.waypoints[:, 1] - y)
        return np.argmin(dists)

    def compute_errors(self, x, y, yaw, closest_idx):
        """
        Compute cross-track error and heading error
        relative to the reference waypoint ahead.
        """
        n = len(self.waypoints)
        ref_idx = (closest_idx + self.LOOKAHEAD_IDX) % n
        ref_x, ref_y = self.waypoints[ref_idx]

        # Heading of path at reference point
        next_idx = (ref_idx + 1) % n
        path_dx = self.waypoints[next_idx, 0] - ref_x
        path_dy = self.waypoints[next_idx, 1] - ref_y
        path_yaw = math.atan2(path_dy, path_dx)

        # Cross-track error — perpendicular distance from path
        dx = x - ref_x
        dy = y - ref_y
        cross_track_error = -math.sin(path_yaw) * dx + math.cos(path_yaw) * dy

        # Heading error — difference in yaw
        heading_error = yaw - path_yaw
        # Normalize to [-pi, pi]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        return cross_track_error, heading_error, ref_idx

    def check_lap(self, x, y):
        dist_to_start = math.hypot(
            x - self.start_point[0],
            y - self.start_point[1])
        if dist_to_start < self.LAP_THRESHOLD:
            if not self.near_start:
                self.near_start = True
                if self.lap_count == 0:
                    self.lap_start   = time.time()
                    self.total_start = time.time()
                    self.lap_count   = 1
                    self.get_logger().info('--- LAP 1 STARTED ---')
                else:
                    lap_time   = time.time() - self.lap_start
                    total_time = time.time() - self.total_start
                    self.lap_count += 1
                    self.get_logger().info(
                        f'✅ LAP {self.lap_count-1} COMPLETE | '
                        f'Lap: {lap_time:.2f}s | Total: {total_time:.2f}s')
                    self.lap_start = time.time()
                    if self.lap_count - 1 == 3:
                        self.get_logger().info(
                            f'🏁 3 LAPS DONE! Total: {total_time:.2f}s | '
                            f'Avg: {total_time/3:.2f}s')
                        stop_msg = AckermannDriveStamped()
                        stop_msg.drive.speed = 0.0
                        stop_msg.drive.steering_angle = 0.0
                        self.pub.publish(stop_msg)
                        self.get_logger().info('🛑 Car stopped!')
                        raise SystemExit
        else:
            self.near_start = False

    def odom_cb(self, msg):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))

        self.check_lap(x, y)

        closest_idx = self.find_closest_waypoint(x, y)
        cte, he, ref_idx = self.compute_errors(x, y, yaw, closest_idx)

        # LQR control law: u = -K * [cte, he]
        error = np.array([cte, he])
        steering = float(-self.K @ error)
        steering = np.clip(steering, -self.MAX_STEER, self.MAX_STEER)

        # Speed from profile
        speed = float(self.speeds[ref_idx] * self.SPEED_GAIN)

        msg_out = AckermannDriveStamped()
        msg_out.drive.speed          = speed
        msg_out.drive.steering_angle = steering
        self.pub.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = LQRController()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()
