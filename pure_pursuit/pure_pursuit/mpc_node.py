import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import csv, math, time
import numpy as np
import cvxpy as cp
from scipy.linalg import solve_continuous_are

class MPCController(Node):
    # --- Vehicle params ---
    WHEELBASE  = 0.3302
    MAX_STEER  = 0.4       # rad
    MAX_SPEED  = 6.5       # m/s
    MIN_SPEED  = 1.5       # m/s

    # --- MPC Horizon ---
    N  = 10                # prediction horizon (steps)
    DT = 0.05             # timestep (s)

    # --- Cost weights ---
    # State cost: [x_err, y_err, yaw_err, speed_err]
    Q  = np.diag([10.0, 10.0, 5.0, 1.0])
    # Terminal cost
    Qf = np.diag([10.0, 10.0, 5.0, 1.0])
    # Input cost: [steering, acceleration]
    R  = np.diag([1.0, 0.5])
    # Input rate cost (smoothness)
    Rd = np.diag([2.0, 1.0])

    # --- Path tracking ---
    LOOKAHEAD_IDX = 8
    LAP_THRESHOLD = 1.0

    WAYPOINT_FILE = '/sim_ws/src/pure_pursuit/waypoints/waypoints_speed.csv'

    def __init__(self):
        super().__init__('mpc_controller')
        self.waypoints, self.speeds = self.load_waypoints()
        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints | '
            f'speed {self.speeds.min():.1f}–{self.speeds.max():.1f} m/s')

        # Previous control inputs (for rate cost)
        self.prev_steer = 0.0
        self.prev_accel = 0.0

        # Current speed estimate
        self.current_speed = 0.0

        # Lap tracking
        self.start_point = self.waypoints[0]
        self.lap_count   = 0
        self.lap_start   = None
        self.total_start = None
        self.near_start  = False

        # Collision tracking
        self.collision_count     = 0
        self.lap_collision_count = 0
        self.lap_collisions      = []
        self.in_collision        = False

        self.sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        self.get_logger().info('MPC controller started!')

    def load_waypoints(self):
        pts, spds = [], []
        with open(self.WAYPOINT_FILE) as f:
            for row in csv.DictReader(f):
                pts.append([float(row['x']), float(row['y'])])
                spds.append(float(row['speed']))
        return np.array(pts), np.array(spds)

    def find_closest(self, x, y):
        dists = np.hypot(
            self.waypoints[:, 0] - x,
            self.waypoints[:, 1] - y)
        return int(np.argmin(dists))

    def get_reference_trajectory(self, closest_idx, yaw, speed):
        """
        Build reference trajectory for N steps ahead.
        State: [x, y, yaw, speed]
        """
        n      = len(self.waypoints)
        ref    = np.zeros((4, self.N + 1))
        d_step = max(1, int(speed * self.DT))  # waypoints to skip per step

        idx = closest_idx
        for i in range(self.N + 1):
            ref[0, i] = self.waypoints[idx, 0]
            ref[1, i] = self.waypoints[idx, 1]
            # Path heading at this waypoint
            next_idx   = (idx + 1) % n
            ref[2, i]  = math.atan2(
                self.waypoints[next_idx, 1] - self.waypoints[idx, 1],
                self.waypoints[next_idx, 0] - self.waypoints[idx, 0])
            ref[3, i]  = self.speeds[idx]
            idx = (idx + d_step) % n

        return ref

    def linearize_model(self, yaw, speed, steer):
        """
        Linearize bicycle kinematic model around current state.
        State: [x, y, yaw, speed]
        Input: [steer, accel]
        """
        A = np.eye(4)
        A[0, 2] = -speed * math.sin(yaw) * self.DT
        A[0, 3] =  math.cos(yaw) * self.DT
        A[1, 2] =  speed * math.cos(yaw) * self.DT
        A[1, 3] =  math.sin(yaw) * self.DT
        A[2, 3] =  math.tan(steer) / self.WHEELBASE * self.DT

        B = np.zeros((4, 2))
        B[2, 0] = speed / (self.WHEELBASE * math.cos(steer)**2) * self.DT
        B[3, 1] = self.DT

        return A, B

    def solve_mpc(self, x0, ref):
        """
        Solve MPC optimization using CVXPY.
        Minimize: sum of state error + input cost + input rate cost
        Subject to: dynamics, steering/speed constraints
        """
        nx = 4   # state dim
        nu = 2   # input dim [steer, accel]

        # Decision variables
        x = cp.Variable((nx, self.N + 1))
        u = cp.Variable((nu, self.N))

        cost        = 0.0
        constraints = [x[:, 0] == x0]

        yaw   = x0[2]
        speed = max(x0[3], 0.5)
        steer = self.prev_steer

        for t in range(self.N):
            # Linearize around reference
            A, B = self.linearize_model(
                ref[2, t], max(ref[3, t], 0.5), steer)

            # State error cost
            state_err = x[:, t] - ref[:, t]
            cost += cp.quad_form(state_err, self.Q)

            # Input cost
            cost += cp.quad_form(u[:, t], self.R)

            # Input rate cost (smooth steering)
            if t == 0:
                prev_u = np.array([self.prev_steer, self.prev_accel])
                cost += cp.quad_form(u[:, t] - prev_u, self.Rd)
            else:
                cost += cp.quad_form(u[:, t] - u[:, t-1], self.Rd)

            # Dynamics constraint
            constraints += [x[:, t+1] == A @ x[:, t] + B @ u[:, t]]

            # Input constraints
            constraints += [
                u[0, t] >= -self.MAX_STEER,
                u[0, t] <=  self.MAX_STEER,
                u[1, t] >= -3.0,   # max decel
                u[1, t] <=  3.0,   # max accel
            ]

            # Speed constraints
            constraints += [
                x[3, t] >= self.MIN_SPEED,
                x[3, t] <= self.MAX_SPEED,
            ]

        # Terminal cost
        cost += cp.quad_form(x[:, self.N] - ref[:, self.N], self.Qf)

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if u.value is not None:
                return u.value[0, 0], u.value[1, 0]
        except Exception as e:
            self.get_logger().warn(f'MPC solve failed: {e}')

        return 0.0, 0.0

    def check_lap(self, x, y):
        dist = math.hypot(x - self.start_point[0], y - self.start_point[1])
        if dist < self.LAP_THRESHOLD:
            if not self.near_start:
                self.near_start = True
                if self.lap_count == 0:
                    self.lap_start   = time.time()
                    self.total_start = time.time()
                    self.lap_count   = 1
                    self.get_logger().info(f'Car Speed: {self.MAX_SPEED} m/s')
                    self.get_logger().info('--- LAP 1 STARTED ---')
                else:
                    lap_time   = time.time() - self.lap_start
                    total_time = time.time() - self.total_start
                    self.lap_collisions.append(self.lap_collision_count)
                    self.lap_collision_count = 0
                    self.lap_count += 1
                    self.get_logger().info(
                        f'✅ LAP {self.lap_count-1} COMPLETE | '
                        f'Lap: {lap_time:.2f}s | Total: {total_time:.2f}s | '
                        f'Collisions: {self.lap_collisions[-1]}')
                    self.lap_start = time.time()
                    if self.lap_count - 1 == 3:
                        total_time = time.time() - self.total_start
                        self.get_logger().info('─' * 50)
                        self.get_logger().info(
                            f'🏁 3 LAPS DONE! Total: {total_time:.2f}s | '
                            f'Avg: {total_time/3:.2f}s')
                        self.get_logger().info('📊 Collision Summary:')
                        for i, c in enumerate(self.lap_collisions):
                            self.get_logger().info(f'   Lap {i+1}: {c} collision(s)')
                        self.get_logger().info(
                            f'   Total: {self.collision_count}')
                        self.get_logger().info('─' * 50)
        else:
            self.near_start = False

    def odom_cb(self, msg):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))
        vx  = msg.twist.twist.linear.x
        vy  = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

        self.check_lap(x, y)

        closest_idx = self.find_closest(x, y)
        ref_idx     = (closest_idx + self.LOOKAHEAD_IDX) % len(self.waypoints)
        ref         = self.get_reference_trajectory(ref_idx, yaw, max(self.current_speed, 0.5))

        # Current state vector
        x0 = np.array([x, y, yaw, self.current_speed])

        # Solve MPC
        steer, accel = self.solve_mpc(x0, ref)

        # Integrate speed
        new_speed = np.clip(
            self.current_speed + accel * self.DT,
            self.MIN_SPEED, self.MAX_SPEED)

        # Update previous inputs
        self.prev_steer = steer
        self.prev_accel = accel

        msg_out = AckermannDriveStamped()
        msg_out.drive.speed          = float(new_speed)
        msg_out.drive.steering_angle = float(steer)
        self.pub.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
