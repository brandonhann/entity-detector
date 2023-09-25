import cv2
import time
import csv
import hashlib

def detect_and_outline_entities(video_path, output_path, fps, skip_interval_seconds=1, count_in_area=True, min_time_threshold=1):
    skip_frames_interval = int(fps * skip_interval_seconds)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    FIXED_WIDTH = 640
    FIXED_HEIGHT = 480

    output_fps = fps / skip_frames_interval
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (FIXED_WIDTH, FIXED_HEIGHT))

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    all_detected_entities = []
    entity_data = {}

    AREA_COLOR = (0, 0, 255)
    AREA_ALPHA = 0.6
    area_x, area_y, area_w, area_h = 500, 100, 400, 200
    area_label = "A1"

    frame_count = 0

    def is_new_entity(x, y, w, h, all_detected_entities, proximity_threshold=30):
        for (old_x, old_y, old_w, old_h, old_frame, old_hash) in all_detected_entities:
            if abs(x - old_x) < proximity_threshold and abs(y - old_y) < proximity_threshold:
                return False, old_hash
        return True, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames_interval == 0:
            boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

            overlay = frame.copy()
            cv2.rectangle(overlay, (area_x, area_y), (area_x+area_w, area_y+area_h), AREA_COLOR, -1)
            cv2.addWeighted(overlay, AREA_ALPHA, frame, 1 - AREA_ALPHA, 0, frame)
            cv2.putText(frame, area_label, (area_x + 10, area_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            current_entities_in_area = 0

            for (x, y, w, h) in boxes:
                new_entity, existing_hash = is_new_entity(x, y, w, h, all_detected_entities)

                if new_entity:
                    entity_str = f"{x},{y},{w},{h},{frame_count}"
                    current_entity_hash = hashlib.md5(entity_str.encode()).hexdigest()
                    all_detected_entities.append((x, y, w, h, frame_count, current_entity_hash))
                    entity_data[current_entity_hash] = {"overall_time": 0, "time_in_red": 0}
                else:
                    current_entity_hash = existing_hash

                entity_data[current_entity_hash]["overall_time"] += skip_interval_seconds
                
                if (x + w > area_x and x < area_x + area_w and y + h > area_y and y < area_y + area_h):
                    entity_data[current_entity_hash]["time_in_red"] += skip_interval_seconds
                    current_entities_in_area += 1

                time_text = f"{entity_data[current_entity_hash]['overall_time']}s"
                cv2.putText(frame, time_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            text_spacing = 50

            cv2.rectangle(frame, (0, frame.shape[0] - text_spacing * 3), (650, frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, f'Current Entities: {len(boxes)}', (10, frame.shape[0] - 2 * text_spacing), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Total Detected: {len(all_detected_entities)}', (10, frame.shape[0] - text_spacing), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Current in Area: {current_entities_in_area}', (10, frame.shape[0]), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)

            resized_frame = cv2.resize(frame, (FIXED_WIDTH, FIXED_HEIGHT))
            out.write(resized_frame)
            cv2.imshow('Entity Detection', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # filter out entities with very short durations (likely false positives)
    filtered_entity_data = {k: v for k, v in entity_data.items() if v["overall_time"] >= min_time_threshold}

    with open('./output/entity_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['Entity ID', 'Overall Time (s)', 'Time in A1 (s)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entity_id, data in filtered_entity_data.items():
            writer.writerow({'Entity ID': entity_id, 
                             'Overall Time (s)': min(data["overall_time"], video_duration), 
                             'Time in A1 (s)': min(data["time_in_red"], data["overall_time"])})

cap_temp = cv2.VideoCapture('video.mp4')
fps = int(cap_temp.get(cv2.CAP_PROP_FPS))
cap_temp.release()

skip_interval_seconds = 0.5
detect_and_outline_entities('video.mp4', './output/result.avi', fps, skip_interval_seconds=skip_interval_seconds, count_in_area=True)
