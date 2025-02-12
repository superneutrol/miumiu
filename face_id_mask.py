import cv2
import os

def capture_images(face_id):
    # Khởi tạo camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Không thể mở camera. Vui lòng kiểm tra lại.")
        return

    # Tạo thư mục dataset nếu chưa tồn tại
    if not os.path.exists('mini_dataset'):
        os.makedirs('mini_dataset')

    # Tải bộ phân loại Haar Cascade
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Đếm số lượng ảnh
    count = 1

    while(True):
        ret, img = cam.read()
        if ret:
            # Chuyển ảnh thành grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Phát hiện khuôn mặt
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                
                # Cắt phần khuôn mặt
                roi_gray = gray[y:y+h, x:x+w]

                # Lưu ảnh
                cv2.imwrite('mini_dataset/User.'+str(face_id) + '.' + str(count) + '.jpg', roi_gray)
                print(f"Đã chụp ảnh thứ {count} cho ID {face_id}")
                
                count += 1

                # Kiểm tra số lượng ảnh đã chụp
                if count > 5:
                    print("Đã đạt giới hạn 5 ảnh cho ID này. Bạn muốn tiếp tục với ID khác không? (y/n)")
                    choice = input()
                    if choice.lower() == 'n':
                        return
                    else:
                        # Nhập ID mới
                        face_id = input('\nNhập ID người dùng mới: ')
                        count = 1  # Reset lại số lượng ảnh

            # Hiển thị ảnh
            cv2.imshow('image', img)

            # Nếu nhấn 'q' thì thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Không thể đọc được khung hình")
            break

    # Giải phóng camera
    cam.release()
    cv2.destroyAllWindows()

# Gọi hàm với ID người dùng
face_id = input('\nNhập ID người dùng: ')
capture_images(face_id)