import cv2

# Encoder
input_video_path = 'C:/Users/kuqit/PycharmProjects/Systems/video/input_video.mp4'  # Ορίζουμε τη διαδρομή του εισερχόμενου βίντεο
output_video_path = 'C:/Users/kuqit/PycharmProjects/Systems/video/output_video.mp4'  # Ορίζουμε τη διαδρομή του εξερχόμενου βίντεο

cap = cv2.VideoCapture(input_video_path)  # Δημιουργούμε ένα αντικείμενο για το διάβασμα του εισερχόμενου βίντεο
fps = cap.get(cv2.CAP_PROP_FPS)  # Παίρνουμε τα καρέ ανά δευτερόλεπτο του βίντεο
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Παίρνουμε το πλάτος των frame του βίντεο
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Παίρνουμε το ύψος των frame του βίντεο

# Δημιουργούμε ένα αντικείμενο VideoWriter για το εξερχόμενο βίντεο
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

ret, frame1 = cap.read()
out.write(frame1)  # Γράφουμε το πρώτο I-frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame.shape != frame1.shape:
        frame = cv2.resize(frame, (frame1.shape[1], frame1.shape[0]))

    # Υπολογίζουμε και εμφανίζουμε το frame του σφάλματος
    error_frame = cv2.absdiff(frame, frame1)
    cv2.imshow("Error Frame", error_frame)
    cv2.waitKey(1)

    # Κωδικοποιούμε το frame του σφάλματος (δεν είναι απαραίτητο για απώλειας ασυμπίεστης συμπίεσης)

    # Ενημερώνουμε το προηγούμενο frame
    frame1 = frame

    # Γράφουμε το τρέχον frame στο εξερχόμενο βίντεο
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()


# Decoder
encoded_video_path = 'C:/Users/kuqit/PycharmProjects/Systems/video/output_video.mp4'  # Ορίζουμε τη διαδρομή του κωδικοποιημένου βίντεο
decoded_output_path = 'C:/Users/kuqit/PycharmProjects/Systems/video/decoded_video.mp4'  # Ορίζουμε τη διαδρομή του αποκωδικοποιημένου βίντεο

cap = cv2.VideoCapture(encoded_video_path)  # Δημιουργούμε ένα αντικείμενο για το διάβασμα του κωδικοποιημένου βίντεο
fps = cap.get(cv2.CAP_PROP_FPS)  # Παίρνουμε τα καρέ ανά δευτερόλεπτο του βίντεο
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Παίρνουμε το πλάτος των frame του βίντεο
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Παίρνουμε το ύψος των frame του βίντεο

# Δημιουργούμε ένα αντικείμενο VideoWriter για το αποκωδικοποιημένο βίντεο
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(decoded_output_path, fourcc, fps, (width, height), isColor=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Αποκωδικοποιούμε το frame (δεν είναι απαραίτητο για απώλειας ασυμπίεστης συμπίεσης)

    # Εμφανίζουμε το αποκωδικοποιημένο frame
    cv2.imshow("Decoded Frame", frame)
    cv2.waitKey(1)

    # Γράφουμε το αποκωδικοποιημένο frame στο εξερχόμενο βίντεο
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()