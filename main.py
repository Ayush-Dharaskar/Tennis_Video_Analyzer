from utils import read_vid, save_vid,draw_player_stats,pixels_to_meters
from trackers import playerTracker, ballTracker,courtLineDetector
from virtual_court import virtualCourt
import cv2
from utils import get_distance
from copy import deepcopy
import pandas as pd
import Dimensions
def main():
    #read video
    vid_dir  = "data/input_video.mp4" 
    vid_frames = read_vid(vid_dir) #(1080, 1920, 3)

    #detect players
    player_tracker = playerTracker(model_path = "yolov8x")
    ball_tracker = ballTracker(model_path = "models/ball_best.pt")
    player_detection = player_tracker.track_multiple_frames(vid_frames,
                                                            use_stubs=True,
                                                            stub_path="tracker_stubs/players.pkl")
    
    ball_detection = ball_tracker.track_multiple_frames(vid_frames,
                                                        use_stubs=True,
                                                        stub_path="tracker_stubs/balls.pkl")
  
    # player_detection = player_tracker.track_multiple_frames(vid_frames[0:5],
    #                                                         use_stubs=False,
    #                                                         )
    
    # ball_detection = ball_tracker.track_multiple_frames(vid_frames[0:5],
    #                                                     use_stubs=False,
    #    
    #  )

    #player_detection and ball detection are a list of dictionaries


    ball_detection = ball_tracker.interpolate_ball(ball_detection)

    #detecting keypoints
    keypoints_model_path = "models/20keypoints_model.pth"
    court_line_detector = courtLineDetector(keypoints_model_path)
    court_markers = court_line_detector.predict_keypoints(vid_frames[0])
    

    #chooseing closest players
    player_detection = player_tracker.filter_players(player_detection,court_markers)
    
    #initilize virtual court
    vc = virtualCourt(vid_frames[0])

    #detecting ball hits
    frames_ball_hit = ball_tracker.get_frames_ball_hit(ball_detection)
    print(frames_ball_hit)
    # Get vc coods
    vc_player_detection,vc_ball_detection = vc.convert_to_vc_coordinates(player_detection,ball_detection,court_markers)
    


    # getting analysis
    players_stats = [{
        'frames': 0,
        'Player_1_shots':0,
        'Player_1_total_shot_speed':0,
        'Player_1_last_shot_speed':0,
        'Player_1_total_player_speed':0,
        'Player_1_last_player_speed':0,
        

        'Player_2_shots':0,
        'Player_2_total_shot_speed':0,
        'Player_2_last_shot_speed':0,
        'Player_2_total_player_speed':0,
        'Player_2_last_player_speed':0,
    }]
    for i in range(len(frames_ball_hit)-1):
        start = frames_ball_hit[i]
        end = frames_ball_hit[i+1]
        #time
        time_between_ball_hit_sec = (end-start)/24 #24fps
        #distance
        distance_covered_ball_pixels = get_distance(vc_ball_detection[start][1],vc_ball_detection[end][1])
        distance_covered_ball_meters = pixels_to_meters(distance_covered_ball_pixels,vc.get_vc_width(),Dimensions.DOUBLES_COURT_WIDTH)  

        #speed
        speed_of_ball = distance_covered_ball_meters/time_between_ball_hit_sec *3.6 #m/s to km/h
        print(f"frame:{start} time:{time_between_ball_hit_sec} distance_covered_ball_meters {distance_covered_ball_meters} speed: {speed_of_ball} km/h")
        #who shot
        players_current = vc_player_detection[start]
        player_shot_ball = min(players_current.keys(),key=lambda x:get_distance(players_current[x],vc_ball_detection[start][1]))    

        #opponent
        opponent_id = 1 if player_shot_ball == 2 else 2
        distance_covered_opponent_pixels = get_distance(vc_player_detection[start][opponent_id],vc_player_detection[end][opponent_id])
        distance_covered_opponent_meters = pixels_to_meters(distance_covered_opponent_pixels,vc.get_vc_width(),Dimensions.DOUBLES_COURT_WIDTH)
        speed_of_opponent = distance_covered_opponent_meters/time_between_ball_hit_sec *3.6

        #setting stats
        current_stats = deepcopy(players_stats[-1])
        current_stats['frames'] = start
        current_stats[f'Player_{player_shot_ball}_shots'] += 1
        current_stats[f'Player_{player_shot_ball}_total_shot_speed'] += speed_of_ball
        current_stats[f'Player_{player_shot_ball}_last_shot_speed'] = speed_of_ball

        current_stats[f'Player_{opponent_id}_total_player_speed'] += speed_of_opponent
        current_stats[f'Player_{opponent_id}_last_player_speed'] = speed_of_opponent

        players_stats.append(current_stats)
    
    df_player_stats = pd.DataFrame(players_stats)
    df_frames = pd.DataFrame({'frames': list(range(len(vid_frames)))})
    
    df_player_stats =pd.merge(df_frames,df_player_stats,on ='frames',how='left')
    df_player_stats = df_player_stats.fillna(method='ffill')
    #df_player_stats = df_player_stats.ffill()
    df_player_stats['Player_1_avg_shot_speed'] = df_player_stats['Player_1_total_shot_speed']/df_player_stats['Player_1_shots']
    df_player_stats['Player_2_avg_shot_speed'] = df_player_stats['Player_2_total_shot_speed']/df_player_stats['Player_2_shots']
    df_player_stats['Player_1_avg_player_speed'] = df_player_stats['Player_1_total_player_speed']/df_player_stats['Player_2_shots']
    df_player_stats['Player_2_avg_player_speed'] = df_player_stats['Player_2_total_player_speed']/df_player_stats['Player_1_shots']

    #Draw Outputs

    #-Player boxes
    output_frames = player_tracker.draw_boxes_on_vid(vid_frames,player_detection)
    
    #-ball box
    output_frames = ball_tracker.draw_boxes_on_vid(output_frames,ball_detection)

    #-keypoints
    output_frames = court_line_detector.draw_keypoints_on_vid(output_frames,court_markers)

    #-virtual court
    output_frames = vc.draw_vc(output_frames)
    output_frames = vc.draw_vc_markers_on_vid(output_frames,vc_player_detection)
    output_frames = vc.draw_vc_markers_on_vid(output_frames,vc_ball_detection,color=(255,255,0))

    #player stats
    output_frames = draw_player_stats(output_frames,df_player_stats)
    #-frame numbers
    for i,frame in enumerate(output_frames):
        cv2.putText(frame,f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    save_vid(output_frames, "data/output/output_video.avi")

    

if __name__ == "__main__":
    main()