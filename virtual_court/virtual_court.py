import cv2
import sys
import numpy as np
sys.path.append("../")
import Dimensions
from utils import(pixels_to_meters,
                  meters_to_pixels,
                  get_foot_bbox,
                  get_closest_marker,
                    get_height_bbox,
                    xy_distance,
                    center_of_bbox,
                    get_distance)

class virtualCourt:
    def __init__(self,frame):
        self.vc_bg_width = 250
        self.vc_bg_height = 500
        self.margin = 50
        self.padding = 20

        self.set_vc_bg(frame)
        self.set_vc_boundary()
        self.set_vc_markers()
        self.set_vc_lines()

    def set_vc_bg(self,frame):
        frame = frame.copy()

        self.x2 = frame.shape[1]-self.margin
        self.x1 = self.x2 - self.vc_bg_width
        self.y2 = self.margin + self.vc_bg_height
        self.y1 = self.margin

    def set_vc_boundary(self):
        self.boundary_x1 = self.x1 + self.padding
        self.boundary_x2 = self.x2 - self.padding
        self.boundary_y1 = self.y1 + self.padding
        self.boundary_y2 = self.y2 - self.padding
        self.boundary_width = self.boundary_x2 - self.boundary_x1

    def fast_meters_to_pixels(self,meters):
        return meters_to_pixels(meters,self.boundary_width,Dimensions.DOUBLES_COURT_WIDTH)
    
    def set_vc_markers(self):
        draw_markers=[0]*28
        #point 0
        draw_markers[0],draw_markers[1] = int(self.boundary_x1),int(self.boundary_y1)
        #point 1
        draw_markers[2],draw_markers[3] = int(self.boundary_x2),int(self.boundary_y1)
        #point 2
        draw_markers[4] = int(self.boundary_x1)
        draw_markers[5] = self.boundary_y1 + self.fast_meters_to_pixels(2*Dimensions.HALF_COURT_HEIGHT)
        
        #point 3
        draw_markers[6] = draw_markers[0] + self.boundary_width
        draw_markers[7] = draw_markers[5]
        #point 4
        draw_markers[8] = draw_markers[0]+self.fast_meters_to_pixels(Dimensions.DOUBLES_ALLEY_WIDTH)
        draw_markers[9] = draw_markers[1]
        #point 5
        draw_markers[10] = draw_markers[8]
        draw_markers[11] = draw_markers[5]
        #point 6
        draw_markers[12] = draw_markers[2]-self.fast_meters_to_pixels(Dimensions.DOUBLES_ALLEY_WIDTH)
        draw_markers[13] = draw_markers[3]
        #point 7
        draw_markers[14] = draw_markers[12]
        draw_markers[15] = draw_markers[7]
        #point 8
        draw_markers[16] = draw_markers[8]
        draw_markers[17] = draw_markers[1]+self.fast_meters_to_pixels(Dimensions.NO_MANS_LAND_HEIGHT)
        #point 9
        draw_markers[18] = draw_markers[12] 
        draw_markers[19] = draw_markers[17]
        #point 10
        draw_markers[20] = draw_markers[10]
        draw_markers[21] = draw_markers[11] - self.fast_meters_to_pixels(Dimensions.NO_MANS_LAND_HEIGHT)
        #point 11
        draw_markers[22] = draw_markers[12]
        draw_markers[23] = draw_markers[21]
        #point 12
        draw_markers[24] = int((draw_markers[16]+draw_markers[18])/2)
        # draw_markers[25] = draw_markers[1] + self.fast_meters_to_pixels(Dimensions.SERVICE_COURT_HEIGHT)
        draw_markers[25] = draw_markers[19]
        #point 13
        draw_markers[26] = draw_markers[24]
        draw_markers[27] = draw_markers[21]

        self.draw_markers = draw_markers  

    def set_vc_lines(self):
        self.vc_court_lines=[
            (0,2),
            (1,3),
            (4,5),
            (6,7),
            (0,1),
            (8,9),
            (10,11),
            (2,3),
            (12,13)
        ]
    
    def draw_vc_markers(self,frame):
        for i in range(0,len(self.draw_markers),2):
            x = int(self.draw_markers[i])
            y = int(self.draw_markers[i+1])
            cv2.circle(frame,(x,y),5,(0,0,255),-1)

        for line in self.vc_court_lines:
            start = (int(self.draw_markers[line[0]*2]),int(self.draw_markers[line[0]*2+1]))
            end = (int(self.draw_markers[line[1]*2]),int(self.draw_markers[line[1]*2+1]))
            cv2.line(frame,start,end,(0,0,0),2)

        #net
        net_start = (int(self.draw_markers[0]),int(self.draw_markers[1]+self.fast_meters_to_pixels(Dimensions.HALF_COURT_HEIGHT)))
        net_end = (int(self.draw_markers[2]),int(self.draw_markers[3]+self.fast_meters_to_pixels(Dimensions.HALF_COURT_HEIGHT)))
        cv2.line(frame,net_start,net_end,(0,255,255),2)
        return frame
    
    def draw_vc_bg(self,frame):
        rect = np.zeros_like(frame,dtype=np.uint8)
        cv2.rectangle(rect,(self.x1,self.y1),(self.x2,self.y2),(255,255,255),-1)
        # mask = cv2.bitwise_not(rect)
        mask = rect.astype(bool)
        
        output_frame = frame.copy()
        output_frame[mask] = cv2.addWeighted(output_frame,0.5,rect,0.5,0)[mask]
        return output_frame        
    
    def draw_vc(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_vc_bg(frame)
            frame = self.draw_vc_markers(frame)
            output_frames.append(frame)
        return output_frames
    
    def get_vc_start(self):
        return (self.boundary_x1,self.boundary_y1)
    
    def get_vc_width(self):
        return self.boundary_width
    
    def get_vc_markers(self):
        return self.draw_markers
    
    def get_vc_coordinates(self,foot_coods,closest_marker,closest_marker_xy,player_height_pixels,player_height_meters):

        x_marker_distance_pixels,y_marker_distance_pixels = xy_distance(foot_coods,closest_marker_xy)

        x_marker_distance_meters = pixels_to_meters(x_marker_distance_pixels,player_height_pixels,player_height_meters)
        y_marker_distance_meters = pixels_to_meters(y_marker_distance_pixels,player_height_pixels,player_height_meters)

        #converting to vc coordinates
        x_vc_distance = self.fast_meters_to_pixels(x_marker_distance_meters)
        y_vc_distance = self.fast_meters_to_pixels(y_marker_distance_meters)

        closest_marker_vc = self.draw_markers[2*closest_marker],self.draw_markers[2*closest_marker+1]
        vc_player_coods = (closest_marker_vc[0]+x_vc_distance,closest_marker_vc[1]+y_vc_distance)
        return vc_player_coods
    
    
    
    def convert_to_vc_coordinates(self,players_bbox,balls_bbox,court_markers):
        player_heights = {1:Dimensions.PLAYER1_HEIGHT,2:Dimensions.PLAYER2_HEIGHT}

        output_players = []
        output_balls = []

        for frame_num,frame_bbox in enumerate(players_bbox):
            ball_bbox = balls_bbox[frame_num][1]
            ball_xy = center_of_bbox(ball_bbox)
            closest_player_to_ball = min(frame_bbox.keys(),key=lambda x:get_distance(center_of_bbox(frame_bbox[x]),ball_xy))

            output_player_bbox = {}
            
            for id,player_bbox in frame_bbox.items():
                foot_coods = get_foot_bbox(player_bbox)
                
                #geting closet marker to player
                closest_marker = get_closest_marker(foot_coods,court_markers,[0,2,12,13])
                closest_marker_xy = court_markers[2*closest_marker],court_markers[2*closest_marker+1]
                #getting player height in pixels
                range_begin = max(0,frame_num-20)
                range_end = min(frame_num+50,len(players_bbox))
                bbox_heights_pixels = [get_height_bbox(players_bbox[i][id]) for i in range(range_begin,range_end)]
                max_player_height_pixels = max(bbox_heights_pixels)
                #getting final vc coordinates
                vc_player_coods = self.get_vc_coordinates(foot_coods,closest_marker,closest_marker_xy,max_player_height_pixels,player_heights[id])
                output_player_bbox[id] = vc_player_coods

                #Doing same for ball
                if closest_player_to_ball == id:
                    closest_marker = get_closest_marker(ball_xy,court_markers,[0,1,2,3,4,5,6,7,8,9,10,12,13])
                    # print(f"Frame: {frame_num} Closest Marker: {closest_marker}")
                    closest_marker_xy = court_markers[2*closest_marker],court_markers[2*closest_marker+1]

                    vc_ball_coods = self.get_vc_coordinates(ball_xy,closest_marker,closest_marker_xy,max_player_height_pixels,player_heights[id])
                    output_balls.append({1:vc_ball_coods}) 
            output_players.append(output_player_bbox)
        
        return output_players,output_balls
    
    def draw_vc_markers_on_vid(self,frames,positions,color = (0,255,0)):
        for frame_num,frame in enumerate(frames):
            for _,position in positions[frame_num].items():
                x,y = position
                cv2.circle(frame,(int(x),int(y)),5,color,-1)
        # for frame in frames:
        #     for id,position in positions.items():
        #         x,y = position
        #         cv2.circle(frame,(int(x),int(y)),5,color,-1)

        return frames