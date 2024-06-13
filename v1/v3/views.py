from django.shortcuts import render,HttpResponse
from .models import UserModel
from .models import Video
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
import csv
from .models import UserRank
from . import models
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import cv2
import math
import csv
import pandas as pd
import ast
import shutil
# Create your views here.


def main(request):
    return render(request, 'v2/main.html')
def login(request):
    # POST 요청이 들어오면 로그인 처리 해줌
    if request.method =='POST':
        username = request.POST.get('username',None)
        password = request.POST.get('password',None)
        me=UserModel.objects.get(username=username)
        me_p=UserModel.objects.get(password=password)
        if username == me.username and password == me_p.password:
            return render(request, 'v2/main.html')
        else:
            return render(request, 'v2/login.html')
    # GET 요청이 들어오면 login form을 담고있는 html을 열어줌
    elif request.method =='GET':
        return render(request, 'v2/login.html')

def password(request):
    return render(request, 'v2/password.html')

def register(request):
    if request.method == "POST":
        username = request.POST.get('username',None)
        password = request.POST.get('password',None)
        password2 = request.POST.get('password2',None)
        if password != password2:
            return render(request, ' v2/register.html')
        else:
            new_user = UserModel()
            new_user.username = username
            new_user.password = password
            new_user.save()
        return render(request, 'v2/login.html')
    elif request.method =='GET':
        return render(request, 'v2/register.html')
def check_username(request):
    if request.method == "GET":
        username = request.GET.get('username', None)
        exists = UserModel.objects.filter(username=username).exists()
        return JsonResponse({'exists': exists})


def main(request):
    return render(request, 'v2/main.html')
def lunge(request):
    return render(request, 'v2/lunge.html')
def pushup(request):
    return render(request, 'v2/pushup.html')
def squat(request):
    return render(request, 'v2/squat.html')
def pullup(request):
    return render(request, 'v2/pullup.html')
def events(request):
    return render(request, 'v2/events.html')
def video_list(request):
    return render(request, 'v2/video_list.html')

def import_csv_to_model(csv_file_path):
    try:
        # CSV 파일을 DataFrame으로 읽기
        df = pd.read_csv(csv_file_path, encoding='euc-kr')
        
        # 'rank' 컬럼에서 '올바른 자세'인 행의 개수 구하기
        count_correct_postures = len(df[df['Posture'] == '올바른 자세'])
        
        # rank_value 계산
        rank_value = count_correct_postures * 10
        
        # rank_value를 문자열로 변환하여 모델에 저장
        UserRank.objects.create(rank=str(rank_value))
        
        return rank_value
    except Exception as e:
        # 예외 처리
        print(f"Error in import_csv_to_model: {e}")
        return None

def result(request):
    # CSV 파일 경로 설정
    csv_file_path = os.path.join("frame_point/lunge/lunge_co/all_points.csv")
    
    # CSV 파일 읽기
    if os.path.exists(csv_file_path):
        try:
            # CSV 파일을 모델에 import하고 rank_value를 받음
            rank_value = import_csv_to_model(csv_file_path)
            
            # CSV 파일을 DataFrame으로 읽기
            df = pd.read_csv(csv_file_path, encoding='euc-kr')
            
            # '올바른 자세'인 행의 posture 컬럼 값 리스트로 가져오기
            correct_postures = df[df['Posture'] == '올바른 자세']['Posture'].tolist()
            
            # '올바르지 않은 자세'인 행의 첫 번째 프레임 가져오기
            incorrect_postures = df[df['Posture'] == '올바르지 않은 자세'].iloc[0]
            
            # 'front' 열 데이터 가져오기
            front = incorrect_postures['front']
            
            if front == 'right':
                knee_angle_right_values = incorrect_postures['knee_angle_right']
                if knee_angle_right_values - 85 > 0:
                    csv_text = '오른쪽 다리를 더 굽혀주세요'
                else:
                    csv_text = '비상!오른쪽 다리가 너무 굽혀졌습니다.\n 오른쪽 무릎에 족적근막염을 초래하여 먼 미래에 골다골증이 발생하는 불상사가 생길 위험이 증가하였습니다 건강을 위하여 주의해주세요.'
            elif front == 'left':
                knee_angle_left_values = incorrect_postures['knee_angle_left']
                if knee_angle_left_values - 85 > 0:
                    csv_text = '왼쪽 다리를 더 굽혀주세요'
                else:
                    csv_text = '왼쪽 다리가 너무 굽혀졌습니다.'
            else:
                csv_text = "올바르지 않은 자세가 감지되었습니다."
        except Exception as e:
            # 예외 처리
            print(f"Error in result view: {e}")
            csv_text = "데이터 처리 중 오류가 발생했습니다."
    else:
        csv_text = "CSV 파일을 찾을 수 없습니다."
    
    # 비디오 리스트 가져오기
    videos = Video.objects.all()
    
    # 템플릿 렌더링
    return render(request, 'v2/result.html', {'videos': videos, 'csv_text': csv_text, 'rank_value': rank_value})

def pushup_video(request):
    return render(request, "v2/events.html")
def squat_video(request):
    return render(request, "v2/events.html")


# lunge_video 함수
def lunge_video(request):
    if request.method == "POST":
        if 'document' in request.FILES:
            document = request.FILES["document"]

            # 변경할 파일 이름 설정
            new_file_name = "lunge_test"

            existing_file_path = os.path.join(settings.MEDIA_ROOT, new_file_name + ".mp4")
            if os.path.exists(existing_file_path):
                os.remove(existing_file_path)

            # 기존 데이터베이스 레코드 삭제
            Video.objects.all().delete()

            # 업로드된 파일의 확장자 가져오기
            file_extension = os.path.splitext(document.name)[1]

            # 새 파일 이름과 확장자 결합
            new_file_name_with_extension = f"{new_file_name}{file_extension}"

            # 새로운 파일 저장
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            filename = fs.save(new_file_name_with_extension, document)

            # 데이터베이스에 정보 저장
            document = Video(title=new_file_name_with_extension, document=filename)
            document.save()

    # 가장 최근의 비디오 가져오기 (하나만 존재할 것이므로)
    latest_video = Video.objects.latest('upload_date') if Video.objects.exists() else None
    lunge_tool() 
    return render(request, "v2/events.html", context={"video": latest_video})

# pullup_video 함수
def pullup_video(request):
    if request.method == "POST":
        if 'document' in request.FILES:
            document = request.FILES["document"]

            # 변경할 파일 이름 설정
            new_file_name = "pullup_test"

            existing_file_path = os.path.join(settings.MEDIA_ROOT, new_file_name + ".mp4")
            if os.path.exists(existing_file_path):
                os.remove(existing_file_path)

            # 기존 데이터베이스 레코드 삭제
            Video.objects.all().delete()

            # 업로드된 파일의 확장자 가져오기
            file_extension = os.path.splitext(document.name)[1]

            # 새 파일 이름과 확장자 결합
            new_file_name_with_extension = f"{new_file_name}{file_extension}"

            # 새로운 파일 저장
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            filename = fs.save(new_file_name_with_extension, document)

            # 데이터베이스에 정보 저장
            document = Video(title=new_file_name_with_extension, document=filename)
            document.save()

    # 가장 최근의 비디오 가져오기 (하나만 존재할 것이므로)
    latest_video = Video.objects.latest('upload_date') if Video.objects.exists() else None
    pullup_tool() 
    return render(request, "v2/events.html", context={"video": latest_video})




# calculate_angle 함수 정의
def lunge_calculate_angle(point_a, point_b, point_c):
    # 세 관절 사이의 각도 계산
    vector_ab = (point_b[0] - point_a[0], point_b[1] - point_a[1])
    vector_bc = (point_c[0] - point_b[0], point_c[1] - point_b[1])

    # 벡터의 길이 계산
    magnitude_ab = math.sqrt(vector_ab[0]**2 + vector_ab[1]**2)
    magnitude_bc = math.sqrt(vector_bc[0]**2 + vector_bc[1]**2)

    # 벡터의 내적 계산
    dot_product = vector_ab[0] * vector_bc[0] + vector_ab[1] * vector_bc[1]

    # 코사인 값 계산
    cosine_theta = dot_product / (magnitude_ab * magnitude_bc)
    
    # 각도 계산
    angle_rad = math.acos(cosine_theta)
    angle_deg = math.degrees(angle_rad)
    return 180 - angle_deg

# 왼쪽 엉덩이와 오른쪽 엉덩이의 y 좌표 비교하여 다리 판별
def detect_front_leg(Lhip_point, Rhip_point):
    if Lhip_point[1] < Rhip_point[1]:
        return "left"  # 왼쪽 다리가 앞으로 나옴
    else:
        return "right"  # 오른쪽 다리가 앞으로 나옴

# 데이터프레임의 한 열에서 y값을 출력하는 함수 정의
def lunge_get_y_value(coord_str):
    # 문자열을 파싱하여 튜플로 변환
    if pd.isna(coord_str) or coord_str == "None":
        return math.nan
    coord_tuple = ast.literal_eval(coord_str)
    # 튜플의 두 번째 요소인 y값을 반환
    return coord_tuple[1]

def lunge_find_max_y_frame(df, column_name):
    y_max = []
    max_y_frames = []
    for index, row in df.iterrows():
        y_value = lunge_get_y_value(row['Point 8'])  # 'Your_Column_Name'은 실제 열의 이름으로 바꿔주세요
        y_max.append(y_value)
        max_y_value = max(y_max)
        
    for index, y_value in enumerate(y_max):
        if y_value == max_y_value:
            max_y_frames.append(df.iloc[index]['Frame Name'])
    return max_y_frames

def video_div(style,folder_path,output_folder):
    # 동영상 파일 형식
    video_formats = ('.mp4', '.avi', '.mov')  # 필요에 따라 확장자 추가
    # 폴더 내의 모든 파일 가져오기
    file_list = os.listdir(folder_path)
    # 동영상 파일 필터링 및 동영상 경로 리스트 만들기
    video_paths = [os.path.join(folder_path, file) for file in file_list if file.lower().endswith(video_formats)]
    # 특정 단어가 포함된 동영상 파일만 선택하기
    video_paths_with_keyword = [path for path in video_paths if style in path.lower()]
    # 이미지 저장을 위한 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 각 동영상 파일별로 처리
    for video_path in video_paths_with_keyword:
        # 동영상 파일 열기
        cap = cv2.VideoCapture(video_path)
        # 동영상의 초당 프레임 수 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 0.5초마다 이미지 저장할 시간 간격
        interval = int(fps * 0.5)
        # 프레임 인덱스 초기화
        frame_index = 0
        # 이미지 저장 시간 초기화
        save_time = 0
        # 프레임 단위로 동영상을 읽어와 이미지로 저장
        while True:
            # 동영상 파일로부터 프레임 읽기
            ret, frame = cap.read()
            # 프레임을 모두 읽었거나 동영상 파일이 종료되면 반복문 탈출
            if not ret:
                break
            # 0.5초마다 이미지 저장
            if frame_index >= save_time:
                # 프레임 이미지 저장
                image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_index}.jpg")
                cv2.imwrite(image_path, frame)
                # 이미지 저장 시간 업데이트
                save_time += interval
            # 다음 프레임 인덱스로 이동
            frame_index += 1
        # 동영상 파일 닫기
        cap.release()

# 런지 Body25
def lunge_tool():
    # 동영상 파일 경로
    folder_path = "media"
    style = 'lunge'
    # 저장할 이미지 폴더 경로
    output_folder = "frame_split/lunge/lunge_co"
    video_div(style,folder_path,output_folder)
    
    BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

    POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                      [11, 24], [22, 24], [23, 24]]

    protoFile_body_25 = "pose_deploy.prototxt"
    weightsFile_body_25 = "pose_iter_584000.caffemodel"

    # 결과를 저장할 CSV 파일 경로
    output_csv_folder = "frame_point/lunge/lunge_co"
    output_csv_file = "frame_point/lunge/lunge_co/all_points.csv"
    # 초록선이 추가된 이미지를 저장할 경로
    output_image_folder = "frame_point/lunge/lunge_co"
    os.makedirs(output_csv_folder, exist_ok=True)

    # 이미지 파일 필터링 및 이미지 경로 리스트 만들기
    folder_path = "frame_split/lunge/lunge_co"
    file_list = os.listdir(folder_path)
    image_paths = [os.path.join(folder_path, file) for file in file_list if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 텍스트 파일 열기 (추가 모드)
    with open(output_csv_file, "w", newline='') as csvfile:
        # CSV writer 생성
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            "Frame Name", "Point 0", "Point 1", "Point 2", "Point 3", "Point 4", "Point 5", "Point 6", "Point 7", "Point 8",
            "Point 9", "Point 10", "Point 11", "Point 12", "Point 13", "Point 14", "Point 15", "Point 16", "Point 17",
            "Point 18", "Point 19", "Point 20", "Point 21", "Point 22", "Point 23", "Point 24","Point 25","knee_angle_left","knee_angle_right"
        ])

        # 이미지 경로마다 반복하여 처리
        for image_path in image_paths:
            points = []
            net = cv2.dnn.readNetFromCaffe(protoFile_body_25, weightsFile_body_25)
            # 프레임 이름 추출
            frame_name = os.path.basename(image_path)
            # 이미지 읽어오기
            frame = cv2.imread(image_path)
            # 입력 이미지의 사이즈 정의
            image_height = 368
            image_width = 368
            frame = cv2.resize(frame, (image_width, image_height))

            # 원본 이미지의 높이, 너비를 받아오기
            frame_height, frame_width = frame.shape[:2]
            # 네트워크에 넣기 위한 전처리
            input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

            # 전처리된 blob 네트워크에 입력
            net.setInput(input_blob)

            # 결과 받아오기
            out = net.forward()
            out_height = out.shape[2]
            out_width = out.shape[3]
            threshold = 0.3

            for i in range(len(BODY_PARTS_BODY_25)):
                # 신체 부위의 confidence map
                prob_map = out[0, i, :, :]

                # 최소값, 최대값, 최소값 위치, 최대값 위치
                min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

                # 원본 이미지에 맞게 포인트 위치 조정
                x = (frame_width * point[0]) / out_width
                x = int(x)
                y = (frame_height * point[1]) / out_height
                y = int(y)
                                                                                                
                if prob > threshold:  # [pointed]
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)
                    points.append((x, y))
                
                else:  # [not pointed]
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)
                    points.append(None)

            # 무릎, 발목, 엉덩이 포인트 추출
            Rknee_point = points[10]
            Lknee_point = points[13]
            Rankle_point = points[11]
            Lankle_point = points[14]
            Rhip_point = points[9]
            Lhip_point = points[12]

            if Rknee_point and Lknee_point and Rankle_point and Lankle_point and Rhip_point and Lhip_point:
                # 좌측 다리 각도 계산
                knee_angle_left = lunge_calculate_angle(Lankle_point, Lknee_point, Lhip_point)
                knee_angle_right = lunge_calculate_angle(Rankle_point, Rknee_point, Rhip_point)

                for pair in POSE_PAIRS_BODY_25:
                    part_a = pair[0]
                    part_b = pair[1]
                    if points[part_a] and points[part_b]:
                        cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 2)
                # 이미지 저장
                output_image_path = os.path.join(output_image_folder, "Images", frame_name)
                cv2.imwrite(output_image_path, frame)

                # 각 프레임의 데이터를 CSV 파일에 추가
                csvwriter.writerow([frame_name] + points + [knee_angle_left, knee_angle_right])

    df = pd.read_csv(output_csv_file, encoding='CP949')

    # 초록선이 추가된 이미지를 저장할 경로
    result_image_folder = "frame_point/lunge/lunge_co/Images_co"
    max_y_result = lunge_find_max_y_frame(df, 'Point 8')
    
    # 이미지 파일 필터링 및 이미지 경로 리스트 만들기
    folder_path = "frame_split/lunge/lunge_co"
    file_list = os.listdir(folder_path)
    image_paths = [os.path.join(folder_path, file) for file in file_list if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for frame_name in max_y_result:
        # 주어진 프레임에 해당하는 데이터프레임 추출
        frame_df = df[df['Frame Name'] == frame_name]
        # 프레임의 관절 포인트 추출
        points = frame_df.iloc[0][1:16]  # 프레임의 관절 포인트는 데이터프레임의 두 번째부터 열에 저장되어 있다고 가정
        
        # 관절 포인트 문자열을 튜플로 변환
        points = [ast.literal_eval(p) if pd.notna(p) and p != "None" else None for p in points]
        
        # 무릎, 발목, 엉덩이 포인트 추출
        Rknee_point = points[10]
        Lknee_point = points[13]
        Rankle_point = points[11]
        Lankle_point = points[14]
        Rhip_point = points[9]
        Lhip_point = points[12]
        
        posture = ""
        
        if Rknee_point and Lknee_point and Rankle_point and Lankle_point and Rhip_point and Lhip_point:
            # 좌측 다리 각도 계산
            knee_angle_left = lunge_calculate_angle(Lankle_point, Lknee_point, Lhip_point)
            knee_angle_right = lunge_calculate_angle(Rankle_point, Rknee_point, Rhip_point)
            
            if (detect_front_leg(Lhip_point, Rhip_point) == "left"):
                if(85 <= knee_angle_left <= 95):
                    posture="올바른 자세"
                else:
                    posture="올바르지 않은 자세"
            else:
                if (85 <= knee_angle_right <= 95):
                    posture="올바른 자세"
                else:
                    posture="올바르지 않은 자세"

            df.loc[df['Frame Name'] == frame_name, 'front'] = detect_front_leg(Lhip_point, Rhip_point)
             # 결과를 데이터프레임에 추가
            df.loc[df['Frame Name'] == frame_name, 'Posture'] = posture
            
            #src_image_path = os.path.join(folder_path, frame_name)
            #dst_image_path = os.path.join(result_image_folder, frame_name)
            #shutil.copyfile(src_image_path, dst_image_path)
            
    # 업데이트된 데이터프레임 저장
    df.to_csv(output_csv_file, index=False, encoding='cp949')
# ----------------------------------------------------------------------#



# pullup 실행 코드
def pullup_calculate_angle(a, b, c):
    try:
        rad1 = math.atan((a[1]-b[1]) / (a[0]-b[0]))
    except ZeroDivisionError:
        rad1 = 0
    try:
        rad2 = math.atan((a[1]-c[1]) / (a[0]-c[0]))
    except ZeroDivisionError:
        rad2 = 0    
    angle = abs((rad1-rad2) * 180/math.pi)
    return angle

def calculate_distance(point_a, point_b):
    x1, y1 = point_a
    x2, y2 = point_b
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def pullup_get_y_value(coord_str):
    if pd.isna(coord_str) or coord_str == "None":
        return math.nan
    # 문자열을 파싱하여 튜플로 변환
    coord_tuple = ast.literal_eval(coord_str)
    # 튜플의 두 번째 요소인 y값을 반환
    return coord_tuple[1]

def pullup_find_min_y_frame(df, column_name):
    y_min = []
    min_y_frames = []
    # 데이터프레임의 한 열에서 y값을 추출하여 출력
    for index, row in df.iterrows():
        y_value = pullup_get_y_value(row['Point 12'])  # 'Your_Column_Name'은 실제 열의 이름으로 바꿔주세요
        y_min.append(y_value)
        min_y_value = min(y_min)
        
    for index, y_value in enumerate(y_min):
        if y_value == min_y_value:
            min_y_frames.append(df.iloc[index]['Frame Name'])
    return min_y_frames

def pullup_tool():
    # 동영상 파일 경로
    folder_path = "media"
    style = 'pullup'
    # 저장할 이미지 폴더 경로
    output_folder = "frame_split/pullup/pullco"
    video_div(style,folder_path,output_folder)

    # MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
    BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                    "Background": 15 }

    POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                    ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                    ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                    ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    
    protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose_iter_160000.caffemodel"
    
    # 결과를 저장할 CSV 파일 경로
    output_csv_folder = "frame_point/pullup/pullco"
    output_csv_file = "frame_point/pullup/pullco/all_points.csv"
    # 초록선이 추가된 이미지를 저장할 경로
    output_image_folder = "frame_point/pullup/pullco"
    os.makedirs(output_csv_folder, exist_ok=True)

    # 이미지 파일 필터링 및 이미지 경로 리스트 만들기
    folder_path = "frame_split/pullup/pullco"
    file_list = os.listdir(folder_path)
    image_paths = [os.path.join(folder_path, file) for file in file_list if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # CSV 파일 열기 (append 모드로)
    with open(output_csv_file, 'w', newline='') as csvfile:
        # CSV writer 생성
        csvwriter = csv.writer(csvfile)
        # 헤더 작성
        csvwriter.writerow(["Frame Name", "Point 0", "Point 1", "Point 2", "Point 3", "Point 4", "Point 5", 
                            "Point 6", "Point 7", "Point 8", "Point 9", "Point 10", "Point 11", "Point 12", 
                            "Point 13", "Point 14", "arm_angle_right","arm_angle_left", "shoulder_length","wrist_length", "Posture"])
        # 이미지 경로마다 반복하여 처리
        for image_path in image_paths:
            points = []
            net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
            # 프레임 이름 추출
            frame_name = os.path.basename(image_path)
            # 이미지 읽어오기
            image = cv2.imread(image_path)
            # frame.shape = 불러온 이미지에서 height, width, color 받아옴
            imageHeight, imageWidth, _ = image.shape
            # network에 넣기위해 전처리
            inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
            # network에 넣어주기
            net.setInput(inpBlob)
            # 결과 받아오기
            output = net.forward()
            # 키포인트 검출시 이미지에 그려줌
            for i in range(0, 15):
                # 해당 신체부위 신뢰도 얻음.
                probMap = output[0, i, :, :]
                # global 최대값 찾기
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                # 원래 이미지에 맞게 점 위치 변경
                x = (imageWidth * point[0]) / output.shape[3]
                y = (imageHeight * point[1]) / output.shape[2]
                # 키포인트 검출한 결과가 0.3보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
                if prob > 0.3 :    
                    points.append((int(x), int(y)))
                else :
                    points.append(None)
                csvwriter.writerow([frame_name] + points)
            
            chest_point = points[BODY_PARTS["Chest"]]
            shoulder_left = points[BODY_PARTS["LShoulder"]]
            shoulder_right = points[BODY_PARTS["RShoulder"]]
            elbow_left = points[BODY_PARTS["LElbow"]]
            elbow_right = points[BODY_PARTS["RElbow"]]
            wrist_left = points[BODY_PARTS["LWrist"]]
            wrist_right = points[BODY_PARTS["RWrist"]]
            knee_left = points[BODY_PARTS["LKnee"]]

            if chest_point and shoulder_left and shoulder_right and elbow_right and elbow_left and wrist_left and wrist_right and knee_left:
                # 무릎 각도 계산 (다리 사이의 각도)
                arm_angle_right = pullup_calculate_angle(chest_point, shoulder_right, elbow_right)
                arm_angle_left = pullup_calculate_angle(chest_point, shoulder_left, elbow_left)
                shoulder_length = calculate_distance(shoulder_right, shoulder_left)
                wrist_length = calculate_distance(wrist_right, wrist_left)
                            
                # 초록 선 그리기
                for pair in POSE_PAIRS:
                    partA = pair[0]
                    partB = pair[1]
                    if points[BODY_PARTS[partA]] and points[BODY_PARTS[partB]]:
                        cv2.line(image, points[BODY_PARTS[partA]], points[BODY_PARTS[partB]], (0, 255, 0), 2)
                # 이미지 저장
                output_image_path = os.path.join(output_image_folder, "Images", frame_name)
                cv2.imwrite(output_image_path, image)

                # 각 프레임의 데이터를 CSV 파일에 추가
                csvwriter.writerow([frame_name] + points + [arm_angle_right,arm_angle_left, shoulder_length,wrist_length])

    df = pd.read_csv(output_csv_file, encoding='CP949')
    result_folder = "frame_point/pullup/pullco"
    min_y_result = pullup_find_min_y_frame(df,'Point 12')

    for frame_name in min_y_result:
        # 주어진 프레임에 해당하는 데이터프레임 추출
        frame_df = df[df['Frame Name'] == frame_name]
        
        # 프레임의 관절 포인트 추출
        points = frame_df.iloc[0][1:16]  # 프레임의 관절 포인트는 데이터프레임의 두 번째부터 열에 저장되어 있다고 가정
        
        # 관절 포인트 문자열을 튜플로 변환
        points = [ast.literal_eval(p) if pd.notna(p) and p != "None" else None for p in points]
        
        chest_point = points[BODY_PARTS["Chest"]]
        shoulder_left = points[BODY_PARTS["LShoulder"]]
        shoulder_right = points[BODY_PARTS["RShoulder"]]
        elbow_left = points[BODY_PARTS["LElbow"]]
        elbow_right = points[BODY_PARTS["RElbow"]]
        wrist_left = points[BODY_PARTS["LWrist"]]
        wrist_right = points[BODY_PARTS["RWrist"]]
        
        
        posture = ""
        if chest_point and shoulder_left and shoulder_right and elbow_right and elbow_left and wrist_left and wrist_right:
            arm_angle_right = pullup_calculate_angle(chest_point, shoulder_right, elbow_right)
            arm_angle_left = pullup_calculate_angle(chest_point, shoulder_left, elbow_left)
            shoulder_length = calculate_distance(shoulder_right, shoulder_left)
            wrist_length = calculate_distance(wrist_right, wrist_left)

            if ((shoulder_length * 2.0) >= wrist_length >= (shoulder_length * 1.5)):
                if ((51 >= arm_angle_right >= 41) & (51 >= arm_angle_left >= 41)):
                    posture = "정확한 자세"
                else:
                    posture = "올바르지 못한 자세"
            else:
                posture = "올바르지 못한 자세"

            # 결과를 데이터프레임에 추가
            df.loc[df['Frame Name'] == frame_name, 'Posture'] = posture
            
    # 업데이트된 데이터프레임 저장
    df.to_csv(output_csv_file, index=False, encoding='CP949')