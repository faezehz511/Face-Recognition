import numpy as np
import cv2

# KNN code
def euclideanDist(X1, X2):
    return np.sqrt(sum((X1 - X2) ** 2))


def knn(X1, querypoint, k=11):
    distance = []
    m = X1.shape[0]

    for i in range(m):
        dist = euclideanDist(querypoint, X1[i])
        distance.append((dist, i))

    distance = sorted(distance)
    distance = distance[:k]

    labels = [your_label for (_, your_label) in distance]
    label_counts = np.bincount(labels)

    pred = np.argmax(label_counts)

    return pred


# Camera initialization
cap = cv2.VideoCapture(0)

# Face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Load your face data
your_face_data = np.load('data/faezeh.npy')

counter = 0  # Counter for tracking "Hello!" prints

while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(faces) == 0:
        print("Can't login")  # No faces detected
        continue

    for face in faces:
        x, y, w, h = face

        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        pred_label = knn(your_face_data, face_section.flatten())

        if pred_label == 0:
            counter += 1
            if counter > 10:
                print("Hello!")  # Your face detected
                cv2.putText(frame, 'Welcome', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            print("Can't login")  # Someone else's face detected

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    cv2.imshow('Face Recognition', frame)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
