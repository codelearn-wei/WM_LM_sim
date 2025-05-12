
import pygame
import math
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
LANE_WIDTH = 3.0
SCALE = 10  # 像素/米
MAIN_LANE_LENGTH = 120  # 主道长度为120米
AUX_LANE_LENGTH = 50   # 辅道长度为50米

def world_to_screen(x, y):
    screen_x = x * SCALE
    screen_y = SCREEN_HEIGHT - y * SCALE
    return int(screen_x), int(screen_y)


class Vehicle:
    def __init__(self, x, y, v, a, heading, yaw_rate, length, width, lane):
        self.x = x
        self.y = y
        self.v = v
        self.a = a
        self.heading = heading
        self.yaw_rate = yaw_rate
        self.length = length
        self.width = width
        self.lane = lane
        self.color = None
        self.name = ""
    
    def update(self, dt):
        self.x += self.v * math.cos(self.heading) * dt
        self.y += self.v * math.sin(self.heading) * dt
        self.heading += self.yaw_rate * dt
        self.v += self.a * dt
        self.v = max(0, min(self.v, 33))  # 限制车速在0-33m/s之间
    
    def draw(self, screen):
        corners = self.get_corners()
        screen_corners = [world_to_screen(x, y) for x, y in corners]
        pygame.draw.polygon(screen, self.color, screen_corners)
        
        font = pygame.font.SysFont('Arial', 12)
        text = font.render(self.name, True, BLACK)
        center_x, center_y = world_to_screen(self.x, self.y)
        screen.blit(text, (center_x - text.get_width() // 2, center_y - text.get_height() // 2))
    
    def get_corners(self):
        l, w = self.length, self.width
        cos_h, sin_h = math.cos(self.heading), math.sin(self.heading)
        
        front_offset_x, front_offset_y = (l/2) * cos_h, (l/2) * sin_h
        rear_offset_x, rear_offset_y = -(l/2) * cos_h, -(l/2) * sin_h
        right_offset_x, right_offset_y = (w/2) * sin_h, -(w/2) * cos_h
        left_offset_x, left_offset_y = -(w/2) * sin_h, (w/2) * cos_h
        
        front_right = (self.x + front_offset_x + right_offset_x, 
                       self.y + front_offset_y + right_offset_y)
        front_left = (self.x + front_offset_x + left_offset_x, 
                      self.y + front_offset_y + left_offset_y)
        rear_left = (self.x + rear_offset_x + left_offset_x, 
                     self.y + rear_offset_y + left_offset_y)
        rear_right = (self.x + rear_offset_x + right_offset_x, 
                      self.y + rear_offset_y + right_offset_y)
        
        return [front_right, front_left, rear_left, rear_right]
    
    def is_out_of_bounds(self, road_bounds):
        corners = self.get_corners()
        
        for x, y in corners:
            is_inside = False
            for bound in road_bounds:
                x_min, y_min, x_max, y_max = bound
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    is_inside = True
                    break
            if not is_inside:
                return True
        return False
