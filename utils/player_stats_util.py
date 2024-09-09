import numpy as np
import cv2

def draw_player_stats(frames,df_player_stats):
    for i,row in df_player_stats.iterrows():
        player_1_shot_speed = row['Player_1_last_shot_speed']
        player_2_shot_speed = row['Player_2_last_shot_speed']
        player_1_speed = row['Player_1_last_player_speed']
        player_2_speed = row['Player_2_last_player_speed']

        avg_player_1_shot_speed = row['Player_1_avg_shot_speed']
        avg_player_2_shot_speed = row['Player_2_avg_shot_speed']
        avg_player_1_speed = row['Player_1_avg_player_speed']
        avg_player_2_speed = row['Player_2_avg_player_speed']

        frame = frames[i]
        shapes = np.zeros_like(frame,np.uint8)
        height = 230
        width = 350

        x1 = frame.shape[1]-width
        y1 = frame.shape[0]-height
        x2 = x1+width
        y2 = y1+height

        overlay = frame.copy()
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(0, 0, 0),-1)
        alpha = 0.5
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)
        frames[i] = frame

        text = "     Player 1     Player 2"
        frames[i] = cv2.putText( frames[i], text, (x1+80, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = "Shot Speed"
        frames[i] = cv2.putText(frames[i], text, (x1+10, y1+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
        frames[i] = cv2.putText(frames[i], text, (x1+130, y1+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Player Speed"
        frames[i] = cv2.putText(frames[i], text, (x1+10, y1+120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        frames[i] = cv2.putText(frames[i], text, (x1+130, y1+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        
        text = "avg. S. Speed"
        frames[i] = cv2.putText(frames[i], text, (x1+10, y1+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
        frames[i] = cv2.putText(frames[i], text, (x1+130, y1+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "avg. P. Speed"
        frames[i] = cv2.putText(frames[i], text, (x1+10, y1+200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
        frames[i] = cv2.putText(frames[i], text, (x1+130, y1+200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frames