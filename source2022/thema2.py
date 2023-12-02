import cv2
import numpy as np

# Διαδρομή προς το αρχείο εισόδου και εξόδου του βίντεο
input_video_path = 'C:/Users/kuqit/PycharmProjects/Systems/video/input_video.mp4'
output_video_path = 'C:/Users/kuqit/PycharmProjects/Systems/video/output_video.mp4'

# Άνοιγμα του αρχείου εισόδου βίντεο
cap = cv2.VideoCapture(input_video_path)

# Λήψη των καρέ ανά δευτερόλεπτο (fps), πλάτους και ύψους του βίντεο
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ορισμός του κωδικοποιητή βίντεο και δημιουργία αντικειμένου VideoWriter για το εξαγωγό βίντεο
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Ανάγνωση του πρώτου καρέ του βίντεο
ret, frame = cap.read()

# Επιλογή περιοχής ενδιαφέροντος (ROI) για την παρακολούθηση
roi = cv2.selectROI(frame, False)
cv2.destroyAllWindows()

# Εξαγωγή του χρώματος της ROI για παρακολούθηση
roi_frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

# Έναρξη επεξεργασίας των καρέ του βίντεο
while True:
    # Ανάγνωση του επόμενου καρέ από το βίντεο
    ret, frame = cap.read()

    # Εάν δεν υπάρχουν περισσότερα καρέ, διακοπή της επανάληψης
    if not ret:
        break

    # Μετατροπή του καρέ στον χώρο χρωμάτων HSV για ανίχνευση του χρώματος
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Ορισμός του εύρους χρώματος για την παρακολούθηση (γκρι αποχρώσεις)
    lower_grey = np.array([0, 0, 50])  # Κάτω όριο γκρι απόχρωσης
    upper_grey = np.array([179, 50, 200])  # Άνω όριο γκρι απόχρωσης

    # Δημιουργία μάσκας για την ανίχνευση του γκρι χρώματος εντός του καθορισμένου εύρους
    mask = cv2.inRange(hsv_frame, lower_grey, upper_grey)

    # Ανεύρεση των περιγράμματων του εντοπισμένου γκρι χρώματος στη μάσκα
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Εύρεση του μεγαλύτερου περιγράμματος (του παρακολουθούμενου αντικειμένου)
        largest_contour = max(contours, key=cv2.contourArea)

        # Λήψη του περιβλήματος περιοχής του μεγαλύτερου περιγράμματος
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Ενημέρωση της θέσης του πλαισίου ROI
        roi = (x, y, w, h)
        roi_frame = frame[y:y + h, x:x + w]

    # Διαγραφή της περιοχής ενδιαφέροντος χρησιμοποιώντας την ενημερωμένη θέση
    mask = np.zeros_like(frame)
    mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 255

    # Εφαρμογή μορφολογικών πράξεων για την κλείσιμο των κενών στη μάσκα
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (35, 35), 0)

    # Αναπλήρωση του εντοπισμένου αντικειμένου χρησιμοποιώντας τη μάσκα
    frame = cv2.inpaint(frame, mask[:, :, 0], 3, cv2.INPAINT_NS)

    # Γράψιμο του τροποποιημένου καρέ στο αρχείο εξόδου
    out.write(frame)

    # Εμφάνιση του τροποποιημένου καρέ του βίντεο
    cv2.imshow('Motified Video', frame)

    # Εάν πατηθεί το πλήκτρο 'q', έξοδος από την επανάληψη
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Απελευθέρωση του αντικειμένου ανάγνωσης και εγγραφής βίντεο
cap.release()
out.release()

# Κλείσιμο όλων των ανοιχτών παραθύρων
cv2.destroyAllWindows()
