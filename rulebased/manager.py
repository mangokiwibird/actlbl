def return_activity(keypoints_frontcam, keypoints_sidecam, previous_keypoints):
    if previous_keypoints is None:
        return "none"
    else:
        front_height = sitting_frontcamera(keypoints_frontcam)
    length_x, length_y, angle = sitting_sidecamera(keypoints_sidecam)
    speed = calculate_speed(previous_keypoints[0], keypoints_frontcam[0])
    walking_speed = 0  # adjusting required
    running_speed = 0  # adjusting required
    max_height = 0.75  # adjusting required
    global preact
    if front_height == None:
        return "none"
    else:
        if speed > walking_speed and front_height > max_height and length_x < length_y:
            return "walking"
        elif speed < walking_speed and front_height < max_height and length_x < length_y:
            return "standing"
        elif speed < walking_speed and front_height < max_height and length_x > length_y and angle[0] > 65 and angle[
            0] < 90 and angle[1] > 65 and angle[1] < 90:
            return "sitting"
        elif speed < walking_speed and front_height < max_height and length_x > length_y and angle[0] < 5 and angle[
            1] < 5:
            return "lying"
        elif preact == "walking" or preact == "running" and speed > running_speed:
            preact = activity
            return "running"
        else:
            return "intermediate acting"