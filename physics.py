import math
import numpy as np

class VolleyballSimulation:
    def __init__(self, h, D, v0, alpha_deg, c, m=0.27, dt=0.005):
        """
        Initialize simulation parameters.
        h: Height of serve (m)
        D: Distance from serve line (m) (Positive = behind line)
        v0: Initial speed (m/s)
        alpha_deg: Angle (degrees)
        c: Drag coefficient factor (kg/m) -> F_drag = c * v^2
        m: Mass of ball (kg), default 0.27 kg
        dt: Time step (s)
        """
        self.h = h
        self.D = D
        self.v0 = v0
        self.alpha = math.radians(alpha_deg)
        self.c = c
        self.m = m
        self.dt = dt
        self.g = 9.81
        
        # Court dimensions
        self.net_x = 0
        self.serve_line_x = -9
        self.net_height = 2.43
        self.court_end_x = 9
        
        # Initial state
        # Player is at serve_line_x - D
        self.x0 = self.serve_line_x - D
        self.y0 = h
        self.vx0 = v0 * math.cos(self.alpha)
        self.vy0 = v0 * math.sin(self.alpha)

    def simulate(self):
        t = 0
        x = self.x0
        y = self.y0
        vx = self.vx0
        vy = self.vy0
        
        trajectory = {'x': [], 'y': [], 't': []}
        
        # Track events
        max_height = -float('inf')
        t_max_height = 0
        cleared_net = False
        hit_net = False
        
        # For "return to initial height"
        # We need to detect when y crosses h downwards
        t_return_h = None
        
        trajectory['x'].append(x)
        trajectory['y'].append(y)
        trajectory['t'].append(t)
        
        while y >= 0:
            # Calculate forces
            v = math.sqrt(vx**2 + vy**2)
            
            ax = -(self.c * v * vx) / self.m
            ay = -self.g - (self.c * v * vy) / self.m
            
            # Euler update
            x += vx * self.dt
            y += vy * self.dt
            vx += ax * self.dt
            vy += ay * self.dt
            t += self.dt
            
            trajectory['x'].append(x)
            trajectory['y'].append(y)
            trajectory['t'].append(t)
            
            # Check Max Height
            if y > max_height:
                max_height = y
                t_max_height = t
            
            # Check Net Clearance
            # We check if we passed the net x-coordinate in this step
            prev_x = trajectory['x'][-2]
            if prev_x < self.net_x and x >= self.net_x:
                if y > self.net_height:
                    cleared_net = True
                else:
                    hit_net = True
            
            # Check return to initial height
            # If we are coming down (vy < 0) and cross h
            if vy < 0 and y <= self.h and t_return_h is None:
                 if max_height > self.h + 0.01: # Tolerance
                     t_return_h = t

        # End of loop (y < 0)
        
        # Analyze results
        x_final = trajectory['x'][-1]
        t_final = t
        
        in_bounds = False
        if cleared_net and not hit_net:
            if 0 < x_final <= self.court_end_x:
                in_bounds = True
        
        # Time from peak to return
        t_peak_to_return = None
        if t_return_h is not None:
            t_peak_to_return = t_return_h - t_max_height

        return {
            'trajectory': trajectory,
            'cleared_net': cleared_net,
            'hit_net': hit_net,
            'in_bounds': in_bounds,
            'x_final': x_final,
            't_final': t_final,
            'max_height': max_height,
            't_max_height': t_max_height,
            't_peak_to_return': t_peak_to_return
        }

if __name__ == "__main__":
    # Test run
    sim = VolleyballSimulation(h=2.5, D=0, v0=20, alpha_deg=10, c=0.005)
    res = sim.simulate()
    print(f"Final X: {res['x_final']:.2f}")
    print(f"Cleared Net: {res['cleared_net']}")
    print(f"In Bounds: {res['in_bounds']}")
