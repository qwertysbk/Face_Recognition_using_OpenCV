import face_recognition as fr
import cv2
import os

faces_path = '/home/mrsbk/Faces'

def get_face_encodings():
    face_names = os.listdir(faces_path)
    face_encodings = []

    for i, name in enumerate(face_names):
        face = fr.load_image_file(os.path.join(faces_path, name))
        face_encodings.append(fr.face_encodings(face)[0])
        face_names[i] = os.path.splitext(name)[0]

    return face_encodings, face_names

face_encodings, face_names = get_face_encodings()

video = cv2.VideoCapture(0)
scl = 2

while True:
    success, image = video.read()

    resized_image = cv2.resize(image, (int(image.shape[1] / scl), int(image.shape[0] / scl)))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    face_locations = fr.face_locations(rgb_image)
    unknown_encodings = fr.face_encodings(rgb_image, face_locations)

    for face_encoding, face_location in zip(unknown_encodings, face_locations):
        result = fr.compare_faces(face_encodings, face_encoding, tolerance=0.4)

        if True in result:
            name = face_names[result.index(True)]

            top, right, bottom, left = face_location

            cv2.rectangle(image, (left * scl, top * scl), (right * scl, bottom * scl), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left * scl, bottom * scl + 20), font, 0.8, (255, 255, 255), 1)

    cv2.imshow("frame", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()