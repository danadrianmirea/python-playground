import pygame
import random
import math
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (40, 40, 40)
GRAY = (80, 80, 80)
LIGHT_GRAY = (180, 180, 180)
ROAD_COLOR = (60, 60, 60)
ROAD_LINE = (255, 255, 200)
SIDEWALK = (100, 100, 100)
CROSSWALK = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 100)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
BLUE = (100, 150, 255)
DARK_BLUE = (0, 0, 150)
PURPLE = (180, 50, 180)
BROWN = (139, 69, 19)

# Road layout
ROAD_Y_TOP = 150
ROAD_Y_BOTTOM = 550
ROAD_HEIGHT = ROAD_Y_BOTTOM - ROAD_Y_TOP
LANE_COUNT = 4  # 2 lanes each direction
LANE_HEIGHT = ROAD_HEIGHT // LANE_COUNT
MEDIAN_Y = ROAD_Y_TOP + ROAD_HEIGHT // 2

# Traffic light positions
INTERSECTION_X = WIDTH // 2
TRAFFIC_LIGHT_Y_TOP = ROAD_Y_TOP - 40
TRAFFIC_LIGHT_Y_BOTTOM = ROAD_Y_BOTTOM + 40

# Vehicle types
VEHICLE_TYPES = {
    'car': {
        'width': 30,
        'height': 18,
        'max_speed': 4.5,
        'acceleration': 0.15,
        'braking': 0.25,
        'color': None,  # random
        'spawn_weight': 60,
    },
    'truck': {
        'width': 40,
        'height': 22,
        'max_speed': 3.0,
        'acceleration': 0.08,
        'braking': 0.18,
        'color': BROWN,
        'spawn_weight': 20,
    },
    'bus': {
        'width': 45,
        'height': 24,
        'max_speed': 2.5,
        'acceleration': 0.06,
        'braking': 0.15,
        'color': ORANGE,
        'spawn_weight': 10,
    },
    'motorcycle': {
        'width': 18,
        'height': 12,
        'max_speed': 5.5,
        'acceleration': 0.20,
        'braking': 0.30,
        'color': None,
        'spawn_weight': 10,
    },
}

VEHICLE_COLORS = [
    (200, 50, 50), (50, 150, 200), (50, 200, 50),
    (200, 200, 50), (200, 100, 50), (150, 50, 200),
    (50, 200, 200), (200, 150, 50), (100, 100, 100),
    (180, 180, 180), (50, 100, 200), (200, 80, 80),
]


class TrafficLight:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction  # 'vertical' or 'horizontal'
        self.state = 'red'  # 'red', 'yellow', 'green'
        self.timer = 0
        self.cycle_time = {
            'red': 90,
            'green': 120,
            'yellow': 30,
        }
        self.state_duration = self.cycle_time[self.state]
    
    def update(self):
        self.timer += 1
        if self.timer >= self.state_duration:
            self.timer = 0
            if self.state == 'red':
                self.state = 'green'
            elif self.state == 'green':
                self.state = 'yellow'
            elif self.state == 'yellow':
                self.state = 'red'
            self.state_duration = self.cycle_time[self.state]
    
    def draw(self, surface):
        # Draw traffic light housing
        housing_rect = pygame.Rect(self.x - 12, self.y - 30, 24, 60)
        pygame.draw.rect(surface, DARK_GRAY, housing_rect)
        pygame.draw.rect(surface, GRAY, housing_rect, 2)
        
        # Draw lights
        light_positions = [(self.x, self.y - 18), (self.x, self.y), (self.x, self.y + 18)]
        colors = [RED, YELLOW, GREEN]
        states = ['red', 'yellow', 'green']
        
        for pos, color, state in zip(light_positions, colors, states):
            if self.state == state:
                pygame.draw.circle(surface, color, pos, 8)
                pygame.draw.circle(surface, WHITE, pos, 8, 1)
            else:
                pygame.draw.circle(surface, (30, 0, 0) if state == 'red' else (30, 30, 0) if state == 'yellow' else (0, 30, 0), pos, 8)
                pygame.draw.circle(surface, GRAY, pos, 8, 1)


class Pedestrian:
    def __init__(self, x, y, target_x, target_y):
        self.x = x
        self.y = y
        self.target_x = target_x
        self.target_y = target_y
        self.speed = random.uniform(0.8, 1.5)
        self.size = 6
        self.color = random.choice([(200, 150, 100), (180, 120, 80), (220, 180, 140), (160, 100, 60)])
        self.alive = True
        
        # Calculate direction
        dx = target_x - x
        dy = target_y - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0:
            self.vx = dx / dist * self.speed
            self.vy = dy / dist * self.speed
        else:
            self.vx = 0
            self.vy = 0
    
    def update(self):
        if not self.alive:
            return
        
        self.x += self.vx
        self.y += self.vy
        
        # Check if reached target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        if abs(dx) < 2 and abs(dy) < 2:
            self.alive = False
    
    def draw(self, surface):
        if not self.alive:
            return
        # Draw pedestrian as a small person
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y - 4)), 3)  # head
        pygame.draw.line(surface, self.color, (int(self.x), int(self.y - 1)), (int(self.x), int(self.y + 5)), 2)  # body
        pygame.draw.line(surface, self.color, (int(self.x), int(self.y)), (int(self.x - 3), int(self.y + 3)), 2)  # left leg
        pygame.draw.line(surface, self.color, (int(self.x), int(self.y)), (int(self.x + 3), int(self.y + 3)), 2)  # right leg


class Vehicle:
    def __init__(self, x, y, lane, direction, vehicle_type='car'):
        self.x = x
        self.y = y
        self.lane = lane
        self.direction = direction  # 1 = right/down, -1 = left/up
        self.type = vehicle_type
        self.specs = VEHICLE_TYPES[vehicle_type]
        
        self.width = self.specs['width']
        self.height = self.specs['height']
        self.max_speed = self.specs['max_speed']
        self.acceleration = self.specs['acceleration']
        self.braking = self.specs['braking']
        
        self.speed = random.uniform(1.0, self.max_speed * 0.6)
        self.target_speed = self.max_speed
        self.color = self.specs['color'] if self.specs['color'] else random.choice(VEHICLE_COLORS)
        
        # Lane change state
        self.lane_change_timer = 0
        self.lane_change_target = None
        self.lane_change_progress = 0
        
        # Indicator state
        self.turning_left = False
        self.turning_right = False
        self.indicator_timer = 0
        
        # Speed limit zone
        self.speed_limit = 5.0
        
        # Distance to vehicle ahead (for debug)
        self.distance_to_ahead = float('inf')
    
    def get_lane_y(self, lane):
        """Get the Y position for a given lane index."""
        return ROAD_Y_TOP + lane * LANE_HEIGHT + LANE_HEIGHT // 2
    
    def start_lane_change(self, direction):
        """Start changing lanes. direction: -1 (up) or 1 (down)."""
        new_lane = self.lane + direction
        if 0 <= new_lane < LANE_COUNT:
            self.lane_change_target = new_lane
            self.lane_change_progress = 0
            self.lane_change_timer = 20  # frames to complete lane change
    
    def update(self, vehicles, traffic_lights, pedestrians):
        # Update lane change
        if self.lane_change_target is not None:
            self.lane_change_progress += 1
            if self.lane_change_progress >= self.lane_change_timer:
                self.lane = self.lane_change_target
                self.lane_change_target = None
                self.lane_change_progress = 0
        
        # Current target Y
        if self.lane_change_target is not None:
            start_y = self.get_lane_y(self.lane)
            end_y = self.get_lane_y(self.lane_change_target)
            t = self.lane_change_progress / self.lane_change_timer
            # Smooth step
            t = t * t * (3 - 2 * t)
            target_y = start_y + (end_y - start_y) * t
        else:
            target_y = self.get_lane_y(self.lane)
        
        # Smoothly move toward lane center
        dy = target_y - self.y
        self.y += dy * 0.15
        
        # Find vehicle ahead
        vehicle_ahead = None
        min_dist = float('inf')
        for other in vehicles:
            if other is self:
                continue
            # Same lane (or same target lane during change)
            other_lane = other.lane_change_target if other.lane_change_target is not None else other.lane
            my_lane = self.lane_change_target if self.lane_change_target is not None else self.lane
            if other_lane == my_lane:
                # Check if ahead in same direction
                if self.direction == 1:  # moving right
                    if other.x > self.x and other.x - self.x < min_dist:
                        min_dist = other.x - self.x
                        vehicle_ahead = other
                else:  # moving left
                    if other.x < self.x and self.x - other.x < min_dist:
                        min_dist = self.x - other.x
                        vehicle_ahead = other
        
        self.distance_to_ahead = min_dist
        
        # Check traffic light ahead
        light_ahead = None
        if self.direction == 1:  # moving right
            if self.x < INTERSECTION_X:
                light_ahead = traffic_lights[0] if traffic_lights[0].direction == 'horizontal' else traffic_lights[1]
        else:  # moving left
            if self.x > INTERSECTION_X:
                light_ahead = traffic_lights[0] if traffic_lights[0].direction == 'horizontal' else traffic_lights[1]
        
        # Check for pedestrians on crosswalk
        pedestrian_nearby = False
        crosswalk_x = INTERSECTION_X
        for ped in pedestrians:
            if not ped.alive:
                continue
            if abs(ped.x - crosswalk_x) < 30 and abs(ped.y - self.y) < 40:
                pedestrian_nearby = True
                break
        
        # Decision making
        desired_speed = self.target_speed
        
        # Speed limit enforcement
        desired_speed = min(desired_speed, self.speed_limit)
        
        # Adjust for vehicle ahead
        if vehicle_ahead and min_dist < 120:
            # Follow distance: 2 seconds rule
            safe_distance = self.speed * 2 * 10  # pixels
            if min_dist < safe_distance:
                # Too close, brake
                target_speed_ahead = vehicle_ahead.speed
                if target_speed_ahead < self.speed:
                    desired_speed = target_speed_ahead * 0.9
                elif min_dist < 30:
                    desired_speed = 0
            elif min_dist < 60:
                desired_speed = min(desired_speed, vehicle_ahead.speed)
        
        # Adjust for traffic light
        if light_ahead:
            dist_to_light = abs(INTERSECTION_X - self.x) if self.direction == 1 else abs(self.x - INTERSECTION_X)
            if dist_to_light < 200:
                if light_ahead.state == 'red':
                    # Stop at intersection
                    stop_x = INTERSECTION_X - 30 if self.direction == 1 else INTERSECTION_X + 30
                    dist_to_stop = abs(stop_x - self.x)
                    if dist_to_stop < 20:
                        desired_speed = 0
                    elif dist_to_stop < 80:
                        # Slow down
                        desired_speed = min(desired_speed, self.speed * 0.5)
                elif light_ahead.state == 'yellow':
                    if dist_to_light < 100:
                        desired_speed = min(desired_speed, self.speed * 0.3)
                    elif dist_to_light < 150:
                        desired_speed = min(desired_speed, self.speed * 0.7)
        
        # Adjust for pedestrians
        if pedestrian_nearby:
            desired_speed = min(desired_speed, 0.5)
        
        # Accelerate or brake
        if self.speed < desired_speed:
            self.speed = min(self.speed + self.acceleration, desired_speed)
        elif self.speed > desired_speed:
            self.speed = max(self.speed - self.braking, desired_speed)
        
        # Ensure minimum speed (don't fully stop unless necessary)
        if self.speed < 0.1 and desired_speed > 0.1:
            self.speed = 0.1
        
        # Move
        self.x += self.speed * self.direction
        
        # Lane change logic
        if self.lane_change_target is None:
            # Try to change lanes if stuck behind slow vehicle
            if vehicle_ahead and min_dist < 60 and self.speed < self.max_speed * 0.5:
                # Check if adjacent lane is clear
                for direction in [-1, 1]:
                    new_lane = self.lane + direction
                    if 0 <= new_lane < LANE_COUNT:
                        # Check for vehicles in target lane
                        lane_clear = True
                        for other in vehicles:
                            if other is self:
                                continue
                            other_lane = other.lane_change_target if other.lane_change_target is not None else other.lane
                            if other_lane == new_lane:
                                gap = abs(other.x - self.x)
                                if gap < 80:
                                    lane_clear = False
                                    break
                        if lane_clear:
                            self.start_lane_change(direction)
                            break
        
        # Update indicator timer
        self.indicator_timer += 1
    
    def draw(self, surface):
        # Draw vehicle body
        rect = pygame.Rect(
            int(self.x - self.width // 2),
            int(self.y - self.height // 2),
            self.width,
            self.height
        )
        
        # Main body
        pygame.draw.rect(surface, self.color, rect, border_radius=3)
        pygame.draw.rect(surface, BLACK, rect, 1, border_radius=3)
        
        # Windshield
        if self.direction == 1:  # moving right
            windshield_rect = pygame.Rect(
                int(self.x + self.width // 4),
                int(self.y - self.height // 3),
                self.width // 4,
                self.height * 2 // 3
            )
        else:
            windshield_rect = pygame.Rect(
                int(self.x - self.width // 2),
                int(self.y - self.height // 3),
                self.width // 4,
                self.height * 2 // 3
            )
        pygame.draw.rect(surface, (150, 200, 255), windshield_rect, border_radius=2)
        
        # Headlights / taillights
        if self.direction == 1:
            # Headlights (front)
            pygame.draw.circle(surface, (255, 255, 200), (int(self.x + self.width // 2 - 2), int(self.y - 4)), 3)
            pygame.draw.circle(surface, (255, 255, 200), (int(self.x + self.width // 2 - 2), int(self.y + 4)), 3)
            # Taillights (back)
            pygame.draw.circle(surface, RED, (int(self.x - self.width // 2 + 2), int(self.y - 4)), 2)
            pygame.draw.circle(surface, RED, (int(self.x - self.width // 2 + 2), int(self.y + 4)), 2)
        else:
            # Headlights (front)
            pygame.draw.circle(surface, (255, 255, 200), (int(self.x - self.width // 2 + 2), int(self.y - 4)), 3)
            pygame.draw.circle(surface, (255, 255, 200), (int(self.x - self.width // 2 + 2), int(self.y + 4)), 3)
            # Taillights (back)
            pygame.draw.circle(surface, RED, (int(self.x + self.width // 2 - 2), int(self.y - 4)), 2)
            pygame.draw.circle(surface, RED, (int(self.x + self.width // 2 - 2), int(self.y + 4)), 2)
        
        # Brake lights (when braking hard)
        if self.speed < 0.5 and self.speed > 0:
            if self.direction == 1:
                pygame.draw.circle(surface, (255, 0, 0), (int(self.x - self.width // 2 + 2), int(self.y - 4)), 3)
                pygame.draw.circle(surface, (255, 0, 0), (int(self.x - self.width // 2 + 2), int(self.y + 4)), 3)
            else:
                pygame.draw.circle(surface, (255, 0, 0), (int(self.x + self.width // 2 - 2), int(self.y - 4)), 3)
                pygame.draw.circle(surface, (255, 0, 0), (int(self.x + self.width // 2 - 2), int(self.y + 4)), 3)
        
        # Turn indicators
        if self.lane_change_target is not None:
            blink = (self.indicator_timer // 10) % 2 == 0
            if blink:
                if self.lane_change_target > self.lane:  # turning down
                    pygame.draw.circle(surface, YELLOW, (int(self.x), int(self.y + self.height // 2 + 2)), 3)
                else:  # turning up
                    pygame.draw.circle(surface, YELLOW, (int(self.x), int(self.y - self.height // 2 - 2)), 3)


class TrafficFlowSimulation:
    def __init__(self):
        self.vehicles = []
        self.pedestrians = []
        self.traffic_lights = []
        self.spawn_timer = 0
        self.pedestrian_spawn_timer = 0
        self.stats = {
            'total_spawned': 0,
            'total_removed': 0,
            'avg_speed': 0,
            'max_speed': 0,
        }
        
        # Create traffic lights
        self.traffic_lights.append(TrafficLight(INTERSECTION_X, TRAFFIC_LIGHT_Y_TOP, 'horizontal'))
        self.traffic_lights.append(TrafficLight(INTERSECTION_X, TRAFFIC_LIGHT_Y_BOTTOM, 'horizontal'))
        
        # Offset the two lights so they're synchronized
        self.traffic_lights[0].state = 'red'
        self.traffic_lights[0].timer = 0
        self.traffic_lights[1].state = 'red'
        self.traffic_lights[1].timer = 0
        
        # Spawn initial vehicles
        for _ in range(30):
            self.spawn_vehicle(random_bias=True)
    
    def get_vehicle_type(self):
        """Randomly select vehicle type based on spawn weights."""
        total_weight = sum(v['spawn_weight'] for v in VEHICLE_TYPES.values())
        roll = random.uniform(0, total_weight)
        cumulative = 0
        for vtype, specs in VEHICLE_TYPES.items():
            cumulative += specs['spawn_weight']
            if roll <= cumulative:
                return vtype
        return 'car'
    
    def spawn_vehicle(self, random_bias=False):
        """Spawn a new vehicle at the edge of the road."""
        if len(self.vehicles) >= 200:
            return
        
        # Choose direction and lane
        direction = random.choice([-1, 1])
        lane = random.randint(0, LANE_COUNT - 1)
        
        # Upper lanes go right, lower lanes go left (or vice versa)
        # Lanes 0-1: right, Lanes 2-3: left
        if lane < LANE_COUNT // 2:
            direction = 1  # right
            x = -50
        else:
            direction = -1  # left
            x = WIDTH + 50
        
        # Don't spawn too close to existing vehicles
        for v in self.vehicles:
            if v.lane == lane and abs(v.x - x) < 150:
                return
        
        vehicle_type = self.get_vehicle_type()
        y = ROAD_Y_TOP + lane * LANE_HEIGHT + LANE_HEIGHT // 2
        
        vehicle = Vehicle(x, y, lane, direction, vehicle_type)
        self.vehicles.append(vehicle)
        self.stats['total_spawned'] += 1
    
    def spawn_pedestrian(self):
        """Spawn a pedestrian at a crosswalk."""
        if len(self.pedestrians) >= 10:
            return
        
        # Spawn from top or bottom sidewalk
        if random.random() < 0.5:
            # From top
            x = INTERSECTION_X + random.randint(-40, 40)
            y = ROAD_Y_TOP - 20
            target_y = ROAD_Y_BOTTOM + 20
        else:
            # From bottom
            x = INTERSECTION_X + random.randint(-40, 40)
            y = ROAD_Y_BOTTOM + 20
            target_y = ROAD_Y_TOP - 20
        
        self.pedestrians.append(Pedestrian(x, y, x, target_y))
    
    def update(self):
        # Update traffic lights
        for light in self.traffic_lights:
            light.update()
        
        # Update vehicles
        for vehicle in self.vehicles[:]:
            vehicle.update(self.vehicles, self.traffic_lights, self.pedestrians)
            
            # Remove vehicles that have left the screen
            if vehicle.x < -100 or vehicle.x > WIDTH + 100:
                self.vehicles.remove(vehicle)
                self.stats['total_removed'] += 1
        
        # Update pedestrians
        for ped in self.pedestrians[:]:
            ped.update()
            if not ped.alive:
                self.pedestrians.remove(ped)
        
        # Spawn new vehicles periodically
        self.spawn_timer += 1
        spawn_interval = max(15, 60 - len(self.vehicles) // 2)
        if self.spawn_timer >= spawn_interval:
            self.spawn_timer = 0
            self.spawn_vehicle()
        
        # Spawn pedestrians periodically
        self.pedestrian_spawn_timer += 1
        if self.pedestrian_spawn_timer >= 180:  # ~3 seconds
            self.pedestrian_spawn_timer = 0
            if random.random() < 0.4:
                self.spawn_pedestrian()
        
        # Update stats
        if self.vehicles:
            speeds = [v.speed for v in self.vehicles]
            self.stats['avg_speed'] = sum(speeds) / len(speeds)
            self.stats['max_speed'] = max(speeds)
    
    def draw(self, surface):
        # Draw road
        road_rect = pygame.Rect(0, ROAD_Y_TOP, WIDTH, ROAD_HEIGHT)
        pygame.draw.rect(surface, ROAD_COLOR, road_rect)
        pygame.draw.rect(surface, GRAY, road_rect, 2)
        
        # Draw lane lines
        for i in range(1, LANE_COUNT):
            y = ROAD_Y_TOP + i * LANE_HEIGHT
            if i == LANE_COUNT // 2:
                # Median (double yellow line)
                for x in range(0, WIDTH, 40):
                    pygame.draw.line(surface, YELLOW, (x, y - 1), (x + 20, y - 1), 2)
                    pygame.draw.line(surface, YELLOW, (x, y + 1), (x + 20, y + 1), 2)
            else:
                # Dashed white line
                for x in range(0, WIDTH, 50):
                    pygame.draw.line(surface, WHITE, (x, y), (x + 25, y), 1)
        
        # Draw direction arrows on road
        font = pygame.font.Font(None, 20)
        for lane in range(LANE_COUNT):
            y = ROAD_Y_TOP + lane * LANE_HEIGHT + LANE_HEIGHT // 2
            if lane < LANE_COUNT // 2:
                arrow = "→"
            else:
                arrow = "←"
            for x in range(100, WIDTH, 200):
                text = font.render(arrow, True, (100, 100, 100))
                surface.blit(text, (x - 8, y - 10))
        
        # Draw sidewalks
        sidewalk_top = pygame.Rect(0, ROAD_Y_TOP - 20, WIDTH, 20)
        sidewalk_bottom = pygame.Rect(0, ROAD_Y_BOTTOM, WIDTH, 20)
        pygame.draw.rect(surface, SIDEWALK, sidewalk_top)
        pygame.draw.rect(surface, SIDEWALK, sidewalk_bottom)
        
        # Draw crosswalk
        for i in range(-4, 5):
            x = INTERSECTION_X + i * 8
            pygame.draw.line(surface, CROSSWALK, (x, ROAD_Y_TOP), (x, ROAD_Y_BOTTOM), 3)
        
        # Draw traffic lights
        for light in self.traffic_lights:
            light.draw(surface)
        
        # Draw vehicles
        for vehicle in self.vehicles:
            vehicle.draw(surface)
        
        # Draw pedestrians
        for ped in self.pedestrians:
            ped.draw(surface)
    
    def draw_ui(self, surface):
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        # Title
        title = font.render("Traffic Flow Simulation", True, WHITE)
        surface.blit(title, (10, 10))
        
        # Instructions
        instructions = [
            "Space: Pause/Resume",
            "C: Clear all vehicles",
            "R: Reset simulation",
            "Click: Spawn vehicle at cursor",
            "Right-click: Spawn pedestrian",
        ]
        for i, text in enumerate(instructions):
            surface.blit(small_font.render(text, True, (200, 200, 200)), (10, 35 + i * 18))
        
        # Stats
        stats_x = WIDTH - 250
        stats_y = 10
        stats_lines = [
            f"Vehicles: {len(self.vehicles)}",
            f"Total Spawned: {self.stats['total_spawned']}",
            f"Total Removed: {self.stats['total_removed']}",
            f"Avg Speed: {self.stats['avg_speed']:.1f} px/frame",
            f"Max Speed: {self.stats['max_speed']:.1f} px/frame",
            f"Pedestrians: {len(self.pedestrians)}",
        ]
        
        # Traffic light status
        light_state = self.traffic_lights[0].state.upper()
        light_color = GREEN if light_state == 'GREEN' else YELLOW if light_state == 'YELLOW' else RED
        stats_lines.append(f"Traffic Light: {light_state}")
        
        for i, text in enumerate(stats_lines):
            color = light_color if 'Traffic Light' in text else WHITE
            surface.blit(small_font.render(text, True, color), (stats_x, stats_y + i * 18))
        
        # Legend
        legend_y = HEIGHT - 100
        surface.blit(small_font.render("Vehicle Types:", True, WHITE), (10, legend_y))
        for i, (vtype, specs) in enumerate(VEHICLE_TYPES.items()):
            color = specs['color'] if specs['color'] else VEHICLE_COLORS[i]
            pygame.draw.rect(surface, color, (10, legend_y + 20 + i * 18, 15, 12))
            surface.blit(small_font.render(f"{vtype}: {specs['max_speed']:.1f} max", True, WHITE), (30, legend_y + 18 + i * 18))


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Traffic Flow Simulation")
    clock = pygame.time.Clock()
    
    simulation = TrafficFlowSimulation()
    paused = False
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_c:
                    simulation.vehicles.clear()
                elif event.key == pygame.K_r:
                    simulation = TrafficFlowSimulation()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Spawn vehicle near click position
                    if ROAD_Y_TOP < event.pos[1] < ROAD_Y_BOTTOM:
                        lane = (event.pos[1] - ROAD_Y_TOP) // LANE_HEIGHT
                        lane = max(0, min(LANE_COUNT - 1, lane))
                        direction = 1 if lane < LANE_COUNT // 2 else -1
                        vehicle_type = simulation.get_vehicle_type()
                        vehicle = Vehicle(event.pos[0], ROAD_Y_TOP + lane * LANE_HEIGHT + LANE_HEIGHT // 2, lane, direction, vehicle_type)
                        simulation.vehicles.append(vehicle)
                        simulation.stats['total_spawned'] += 1
                elif event.button == 3:  # Right click
                    # Spawn pedestrian
                    ped = Pedestrian(event.pos[0], event.pos[1], event.pos[0], event.pos[1] + 100)
                    simulation.pedestrians.append(ped)
        
        if not paused:
            simulation.update()
        
        screen.fill(BLACK)
        simulation.draw(screen)
        simulation.draw_ui(screen)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()