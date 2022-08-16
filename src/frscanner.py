import cv2
import numpy as np
import face_recognition

def getVideoCaptureSettings(vcap):
    width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    fps =  vcap.get(cv2.CAP_PROP_FPS)

    return dict({'width': width, 'height': height, 'fps': fps})

# get the first available video capture device
video_capture = cv2.VideoCapture(0)

# kirill sample image
kirill_image = face_recognition.load_image_file("../data/kirill.png")
kirill_face_encoding = face_recognition.face_encodings(kirill_image)[0]

# known faces lists
known_face_encodings = [
    kirill_face_encoding,
]

known_face_names = [
    "Kirill",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_frame = True

while True:
    # get frame
    ret, frame = video_capture.read()

    # only process every other frame of video to save time
    if process_frame:
        # resize frame for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # convert from BGR(opencv format) to RGB(face_recognition format)
        rgb_frame = small_frame[:, :, ::-1]

        # find all faces and get face encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # check face matching
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            break

    process_frame = not process_frame

    # display results
    cap_settings = getVideoCaptureSettings(video_capture)
    ellipse_color = (0, 0, 255)
    ellipse_size = (int(cap_settings['width']/6), int(cap_settings['height']/3.5))
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # scale back up face locations since the frame we detected was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # ellipse variables
        face_bbox_width = right - left
        face_bbox_coord_x = (left, right)
        face_bbox_coord_y = (top, bottom)
        ellipse_color = (0, 255, 0) \
            if face_bbox_width < ellipse_size[0] + 150 \
            and face_bbox_coord_x[0] > cap_settings['width']/4 \
            and face_bbox_coord_x[1] < cap_settings['width'] * 3/4 \
            and face_bbox_coord_y[0] > cap_settings['height']/4 \
            and face_bbox_coord_y[1] < cap_settings['height'] * 3/4 \
            else (0, 0, 255)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        if ellipse_color == (0, 255, 0):
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        break

    # ellipse data
    ellipse_origin = (int(cap_settings['width']/2), int(cap_settings['height']/2))
    
    # draw ellipse
    frame = cv2.ellipse(
        frame,
        ellipse_origin,
        ellipse_size,
        0, 0, 360,     # ange, startAngle, endAngle 
        ellipse_color, # color
        3              # thickness
    )

    # display frame
    cv2.imshow('FRScanner', frame)

    # exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clean up
video_capture.release()
cv2.destroyAllWindows()
