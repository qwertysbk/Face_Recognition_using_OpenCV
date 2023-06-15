import face_recognition as fr
import cv2
import os

faces_path = 'Test_Images_Folder_Address' 	# Path of the folder containing the test faces

def get_face_encodings():
    face_names = os.listdir(faces_path)
    face_encodings = []

    # Loop through each face image in the folder
    for i, name in enumerate(face_names):
        face = fr.load_image_file(os.path.join(faces_path, name))
        # Encode the face and store the encoding in a list
        face_encodings.append(fr.face_encodings(face)[0])
        # Store the name of the face (without the file extension) in the face_names list
        face_names[i] = os.path.splitext(name)[0]

    return face_encodings, face_names

# Get the face encodings and names from the test faces
face_encodings, face_names = get_face_encodings()

video = cv2.VideoCapture(0)
scl = 2

while True:
    success, image = video.read()

    # Resize the image for faster processing
    resized_image = cv2.resize(image, (int(image.shape[1] / scl), int(image.shape[0] / scl)))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Locate faces in the resized image
    face_locations = fr.face_locations(rgb_image)
    # Encode the detected faces in the resized image
    unknown_encodings = fr.face_encodings(rgb_image, face_locations)

    for face_encoding, face_location in zip(unknown_encodings, face_locations):
        # Compare the unknown face encodings with the known face encodings
        result = fr.compare_faces(face_encodings, face_encoding, tolerance=0.4)

        if True in result:
            # Find the index of the matched face in the face_encodings list
            index = result.index(True)
            name = face_names[index]

            top, right, bottom, left = face_location

            # Draw a rectangle around the detected face in the original image
            cv2.rectangle(image, (left * scl, top * scl), (right * scl, bottom * scl), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_DUPLEX
            # Put the name of the recognized person below the face rectangle
            cv2.putText(image, name, (left * scl, bottom * scl + 20), font, 0.8, (255, 255, 255), 1)

    # Display the frame with faces and recognized names
    cv2.imshow("frame", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
