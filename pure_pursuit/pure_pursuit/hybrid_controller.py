import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import csv, math, time
import numpy as np

class HybridController(Node):

    # --- Vehicle params ---
    WHEELBASE  = 0.3302
    MAX_STEER  = 0.4

    # --- Pure Pursuit params ---
    LOOKAHEAD     = 1.5
    SPEED_GAIN    = 1.0
    WAYPOINT_FILE = '/sim_ws/src/pure_pursuit/waypoints/waypoints_speed.csv'

    # --- Wall Follower (Reactive) Params ---
    WF_KP           = 1.2
    WF_KD           = 0.09
    WF_DESIRED_DIST = 1.0
    WF_LOOKAHEAD    = 1.0
    WF_SPEED_CORNER = 2.5    # Base speed in corners

    # --- Mode switching thresholds ---
    LIDAR_DANGER_DIST  = 0.4   # metres — switch to reactive if anything closer
    CORNER_CURVATURE   = 0.15  # path curvature threshold to detect corner
    LOOKAHEAD_IDX_CURV = 15    # waypoints ahead to compute curvature

    # --- Lap tracking ---
    LAP_THRESHOLD  = 1.0
    COLLISION_DIST = 0.25

    MODE_PURSUIT  = 'PURE_PURSUIT'
    MODE_REACTIVE = 'REACTIVE'

    def __init__(self):
        super().__init__('hybrid_controller')

        # Load waypoints
        self.waypoints, self.speeds = self.load_waypoints()
        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints | '
            f'speed {self.speeds.min():.1f}-{self.speeds.max():.1f} m/s')

        # State
        self.mode          = self.MODE_PURSUIT
        self.position_x    = 0.0
        self.position_y    = 0.0
        self.yaw           = 0.0
        self.current_speed = 0.0
        self.latest_scan   = None

        # Wall Follower State
        self.wf_prev_error = 0.0
        self.active_wall   = None  # 'LEFT' or 'RIGHT'

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

        # Mode stats
        self.pursuit_time  = 0.0
        self.reactive_time = 0.0
        self.last_mode_time = time.time()

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_cb, 10)

        # Publisher
        self.pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        self.get_logger().info('Hybrid controller (Pure Pursuit + Wall Follower) started!')

    # ─────────────────────────────────────────
    # Waypoint loading
    # ─────────────────────────────────────────
    def load_waypoints(self):
        pts, spds = [], []
        with open(self.WAYPOINT_FILE) as f:
            for row in csv.DictReader(f):
                pts.append([float(row['x']), float(row['y'])])
                spds.append(float(row['speed']))
        return np.array(pts), np.array(spds)

    # ─────────────────────────────────────────
    # Odometry callback
    # ─────────────────────────────────────────
    def odom_cb(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

        if self.start_x is None:
            self.start_x = self.position_x
            self.start_y = self.position_y

        self.check_lap(self.position_x, self.position_y)

    # ─────────────────────────────────────────
    # LiDAR callback — main control loop
    # ─────────────────────────────────────────
    def scan_cb(self, scan):
        self.latest_scan = scan
        self.check_collision(scan)

        # Decide mode
        new_mode = self.select_mode(scan)
        if new_mode != self.mode:
            now = time.time()
            elapsed = now - self.last_mode_time
            if self.mode == self.MODE_PURSUIT:
                self.pursuit_time += elapsed
            else:
                self.reactive_time += elapsed
            self.last_mode_time = now
            self.mode = new_mode
            self.get_logger().info(f'Mode → {self.mode}')

        # Run selected controller
        if self.mode == self.MODE_PURSUIT:
            self.active_wall = None # Reset wall choice for the next corner
            steer, speed = self.pure_pursuit()
        else:
            steer, speed = self.wall_follower(scan)

        # Publish
        msg_out = AckermannDriveStamped()
        msg_out.drive.speed          = float(speed)
        msg_out.drive.steering_angle = float(steer)
        self.pub.publish(msg_out)

    # ─────────────────────────────────────────
    # Mode selector
    # ─────────────────────────────────────────
    def select_mode(self, scan):
        # 1. Check LiDAR for nearby obstacles
        ranges = np.array(scan.ranges)
        ranges = ranges[np.isfinite(ranges)]
        if len(ranges) > 0:
            min_dist = np.min(ranges)
            if min_dist < self.LIDAR_DANGER_DIST:
                return self.MODE_REACTIVE

        # 2. Check path curvature ahead
        if self.start_x is not None:
            curvature = self.compute_path_curvature()
            if curvature > self.CORNER_CURVATURE:
                return self.MODE_REACTIVE

        return self.MODE_PURSUIT

    # ─────────────────────────────────────────
    # Curvature of path ahead
    # ─────────────────────────────────────────
    def compute_path_curvature(self):
        closest = self.find_closest_waypoint()
        n = len(self.waypoints)
        i0 = closest
        i1 = (closest + self.LOOKAHEAD_IDX_CURV // 2) % n
        i2 = (closest + self.LOOKAHEAD_IDX_CURV) % n

        p0 = self.waypoints[i0]
        p1 = self.waypoints[i1]
        p2 = self.waypoints[i2]

        d01 = np.linalg.norm(p1 - p0)
        d12 = np.linalg.norm(p2 - p1)
        d02 = np.linalg.norm(p2 - p0)

        cross = abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
        denom = d01 * d12 * d02

        if denom < 1e-6:
            return 0.0
        return cross / denom

    # ─────────────────────────────────────────
    # Pure Pursuit controller
    # ─────────────────────────────────────────
    def pure_pursuit(self):
        x, y, yaw = self.position_x, self.position_y, self.yaw

        target_idx, target_pt = self.find_lookahead_point(x, y)
        if target_pt is None:
            return 0.0, 1.0

        dx      = target_pt[0] - x
        dy      = target_pt[1] - y
        local_y = -math.sin(yaw)*dx + math.cos(yaw)*dy

        curvature = 2.0 * local_y / (self.LOOKAHEAD ** 2)
        steering  = math.atan(self.WHEELBASE * curvature)
        steering  = np.clip(steering, -self.MAX_STEER, self.MAX_STEER)
        speed     = float(self.speeds[target_idx] * self.SPEED_GAIN)

        return steering, speed

    # ─────────────────────────────────────────
    # Dynamic Wall Follower (Replaces Gap Follower)
    # ─────────────────────────────────────────
    def get_range(self, scan, angle_deg):
        """Safely extracts the LiDAR distance at a specific degree angle."""
        angle_rad = np.radians(angle_deg)
        idx = int((angle_rad - scan.angle_min) / scan.angle_increment)
        idx = np.clip(idx, 0, len(scan.ranges) - 1)
        r = scan.ranges[idx]
        if np.isinf(r) or np.isnan(r):
            return scan.range_max
        return r

    def wall_follower(self, scan):
        # 1. Dynamically pick a wall if we just entered the corner
        if self.active_wall is None:
            dist_right = self.get_range(scan, -90)
            dist_left  = self.get_range(scan, 90)
            self.active_wall = 'RIGHT' if dist_right < dist_left else 'LEFT'

        # 2. Extract specific rays based on the active wall
        theta = np.radians(45)
        if self.active_wall == 'RIGHT':
            a = self.get_range(scan, -45)
            b = self.get_range(scan, -90)
            direction_multiplier = 1.0
        else:
            a = self.get_range(scan, 45)
            b = self.get_range(scan, 90)
            direction_multiplier = -1.0  # Steer right if left wall is too close

        # 3. Calculate Distance Math
        alpha = np.arctan2(a * np.cos(theta) - b, a * np.sin(theta))
        dist_now   = b * np.cos(alpha)
        dist_ahead = dist_now + self.WF_LOOKAHEAD * np.sin(alpha)

        # 4. PD Control
        error = self.WF_DESIRED_DIST - dist_ahead
        derivative = error - self.wf_prev_error
        steering = self.WF_KP * error + self.WF_KD * derivative
        
        # Apply direction logic and clip
        steering = steering * direction_multiplier
        steering = np.clip(steering, -self.MAX_STEER, self.MAX_STEER)
        
        self.wf_prev_error = error

        # 5. Dynamic Speed Scaling (Slow down on tight corners)
        speed = self.WF_SPEED_CORNER * (1.0 - 0.5 * abs(steering) / self.MAX_STEER)

        return steering, speed

    # ─────────────────────────────────────────
    # Waypoint helpers
    # ─────────────────────────────────────────
    def find_closest_waypoint(self):
        dists = np.hypot(
            self.waypoints[:, 0] - self.position_x,
            self.waypoints[:, 1] - self.position_y)
        return int(np.argmin(dists))

    def find_lookahead_point(self, x, y):
        dists   = np.hypot(
            self.waypoints[:, 0] - x,
            self.waypoints[:, 1] - y)
        closest = np.argmin(dists)
        n       = len(self.waypoints)
        for i in range(n):
            idx = (closest + i) % n
            if dists[idx] >= self.LOOKAHEAD:
                return idx, self.waypoints[idx]
        fallback = (closest + 10) % n
        return fallback, self.waypoints[fallback]

    # ─────────────────────────────────────────
    # Collision detection
    # ─────────────────────────────────────────
    def check_collision(self, scan):
        ranges = np.array(scan.ranges)
        ranges = ranges[np.isfinite(ranges)]
        if len(ranges) == 0:
            return
        if np.min(ranges) < self.COLLISION_DIST:
            if not self.in_collision:
                self.in_collision         = True
                self.collision_count     += 1
                self.lap_collision_count += 1
                self.get_logger().info(
                    f'COLLISION! Lap {max(self.lap_count,1)} | '
                    f'Lap: {self.lap_collision_count} | '
                    f'Total: {self.collision_count}')
        else:
            self.in_collision = False

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
                        f'Collisions: {self.lap_collisions[-1]}')
                    self.lap_start = time.time()

                    if self.lap_count - 1 == 3:
                        total_time = time.time() - self.total_start
                        p_pct = self.pursuit_time / (self.pursuit_time + self.reactive_time + 1e-6) * 100
                        r_pct = 100 - p_pct
                        self.get_logger().info('─' * 50)
                        self.get_logger().info(
                            f'3 LAPS DONE! Total: {total_time:.2f}s | '
                            f'Avg: {total_time/3:.2f}s')
                        self.get_logger().info('Collision summary:')
                        for i, c in enumerate(self.lap_collisions):
                            self.get_logger().info(
                                f'  Lap {i+1}: {c} collision(s)')
                        self.get_logger().info(
                            f'  Total collisions: {self.collision_count}')
                        self.get_logger().info(
                            f'Mode time: Pure Pursuit {p_pct:.1f}% | '
                            f'Reactive {r_pct:.1f}%')
                        self.get_logger().info('─' * 50)
        else:
            self.near_start = False

def main(args=None):
    rclpy.init(args=args)
    node = HybridController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
