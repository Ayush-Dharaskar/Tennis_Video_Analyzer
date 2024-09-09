from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import sys
sys.path.append("../")
from utils import (get_distance,
                   center_of_bbox)

class playerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def filter_players(self,players,court_markers):
        first_frame_players = players[0]
        closest_players = self.choose_closest_players(first_frame_players,court_markers)
        filtered_players_detections=[]
        for player in players: #for each frame
            filtered_players = {id:coods for id,coods in player.items() if id in closest_players}
            filtered_players_detections.append(filtered_players)
        return filtered_players_detections

    def choose_closest_players(self,players,court_markers): #returns a list of 2 closest players ids,used for 1 frame
        distances=[]
        for id,player in players.items():
            player_center = center_of_bbox(player)
            min = float("inf")

            for marker in range(0,len(court_markers),2):
                xy_marker =(court_markers[marker],court_markers[marker+1])
                distance = get_distance(player_center,xy_marker)
                if distance < min:
                    min = distance

            distances.append((id,min))

        #sorting players based on distance
        distances = sorted(distances,key=lambda x:x[1])
        closest_players = [distances[0][0],distances[1][0]]
        return closest_players
            
    def track_multiple_frames(self,frames,use_stubs=False,stub_path=None):
        players = []

        if use_stubs == True:
            with open(stub_path,"rb") as f:
                players = pickle.load(f)
            return players
        
        for frame in frames:
            player = self.track_frame(frame)
            players.append(player)

        if stub_path is not None:
            with open(stub_path,"wb") as f:
                pickle.dump(players,f)
        return players
    

    # def track_frame(self,frame):
    #     output = self.model(frame)[0]#persist=True so that the model does not close after the first frame
    #     id_name = output.names 
    #     print(id_name)

    #     for box in output.boxes:
    #         print(box.id)

    #     players={}
    #     for box in output.boxes:
    #         id = int(box.id.tolist()[0])
    #         coods = box.xyxy.tolist()[0]
    #         object_id = box.cls.tolist()[0]
    #         object_name = id_name[object_id]
    #         if object_name == "person":
    #             players[id] = coods
        
    #     return players

    def track_frame(self, frame):
        output = self.model.track(frame,persist=True)[0]  #[0] for models first output
        id_name = output.names


        players = {}
        player_id_counter = 1  # Counter to assign unique IDs to players

        for box in output.boxes:
            coods = box.xyxy.tolist()[0]
            object_id = box.cls.tolist()[0]
            object_name = id_name[object_id]

            # Only consider objects labeled as "person"
            if object_name == "person":
                # Assign a unique ID to the player
                player_id = player_id_counter
                player_id_counter += 1

                # Save the player's coordinates along with their ID
                players[player_id] = coods

        return players
    
    def draw_boxes_on_vid(self,frames,players):
        output_frames = []
        for frame,player in zip(frames,players):
            for id,coods in player.items():
                x1,y1,x2,y2 = coods
                
                cv2.putText(frame,f"player ID: {id}",(int(x1),int(y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)

            output_frames.append(frame)
        return output_frames


