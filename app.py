import streamlit as st
import cv2
import tempfile
from Analyzer import FaceAnalyzer
import os
import io
import uuid
from PIL import Image

def init_session():
    st.session_state.setdefault('camera_running', False)
    st.session_state.setdefault('model_path')
    st.session_state.setdefault('face_match_threshold', 0.6)
    st.session_state.setdefault('max_faces', 5)
    st.session_state.setdefault('frame_skip', 2)
    st.session_state.setdefault('temp_faces', {})
    st.session_state.setdefault('unknown_faces', {})

def create_analyzer():
    return FaceAnalyzer(
        model_path=st.session_state.model_path,
        temp_faces=st.session_state.temp_faces
    )

def configure_analyzer(analyzer):
    analyzer.FACE_MATCH_THRESHOLD = st.session_state.face_match_threshold
    analyzer.MAX_FACES = st.session_state.max_faces
    analyzer.FRAME_SKIP = st.session_state.frame_skip
    return analyzer

def process_frame(analyzer, frame, frame_placeholder):
    processed = analyzer.process_frame(frame)
    frame_placeholder.image(
        cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
        channels="RGB"
    )

def handle_unknown_faces(analyzer, frame, is_camera):
    if hasattr(analyzer, 'detect_unknown_faces'):
        unknown_faces = analyzer.detect_unknown_faces(frame)
    else:
        unknown_faces = getattr(analyzer, 'unknown_faces', [])
    
    if is_camera and len(unknown_faces) > 0:
        st.write(f"Found {len(unknown_faces)} unknown faces")
        
    store_unknown_faces(unknown_faces)

def store_unknown_faces(unknown_faces):
    for face_img in unknown_faces:
        face_id = str(uuid.uuid4())[:8]
        is_success, buffer = cv2.imencode(".jpg", face_img)
        if is_success:
            st.session_state.unknown_faces[face_id] = buffer.tobytes()

def check_stop_condition(is_camera):
    if not is_camera and st.session_state.get('stop_video', False):
        return True
    if is_camera and not st.session_state.camera_running:
        return True
    return False

def handle_media_stream(input_source, is_camera=True):
    analyzer = create_analyzer()
    analyzer = configure_analyzer(analyzer)
    frame_placeholder = st.empty()
    
    cap = cv2.VideoCapture(input_source if is_camera else str(input_source))
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            process_frame(analyzer, frame, frame_placeholder)
            handle_unknown_faces(analyzer, frame, is_camera)
            
            if check_stop_condition(is_camera):
                break
            
    finally:
        cap.release()
    return analyzer

def display_model_settings():
    st.sidebar.header("⚙️ FaceAnalyzer Ayarları")

    st.session_state.model_path = st.sidebar.radio(
        "Model Seçimi",
        ("src/yolov11l-face.pt", "src/yolov8n-face.pt"),
        index=0,
        format_func=lambda x: x.split(".")[0].split("/")[-1]
    )
    st.sidebar.slider("Face Match Threshold", 0.3, 1.0, value=st.session_state.face_match_threshold, key="face_match_threshold")
    st.sidebar.slider("Max Faces", 1, 10, value=st.session_state.max_faces, key="max_faces")
    st.sidebar.slider("Frame Skip", 1, 10, value=st.session_state.frame_skip, key="frame_skip")

def display_saved_face(face_name):
    col1, col2 = st.sidebar.columns([1, 3])
    with col1:
        st.image(io.BytesIO(st.session_state.temp_faces[face_name]), width=60)
    with col2:
        st.write(face_name)
        if st.button(f"❌ {face_name}", key=f"del_{face_name}"):
            del st.session_state.temp_faces[face_name]
            st.rerun()

def add_new_face_ui():
    st.sidebar.markdown("**Yeni Yüz Ekle**")
    uploaded_face = st.sidebar.file_uploader("Yüz fotoğrafı yükle", type=["jpg", "jpeg", "png"])
    new_name = st.sidebar.text_input("Yüz ismi")
    
    if st.sidebar.button("Yüz Ekle") and uploaded_face and new_name:
        try:
            st.session_state.temp_faces[new_name.strip()] = uploaded_face.getvalue()
            st.success(f"{new_name.strip()} eklendi!")
            st.rerun()
        except Exception as e:
            st.error(f"Hata: {e}")

def settings_interface():
    display_model_settings()
    st.sidebar.markdown("---")
    st.sidebar.subheader("Geçici Yüzler")

    if st.session_state.temp_faces:
        for face_name in list(st.session_state.temp_faces.keys()):
            display_saved_face(face_name)
    else:
        st.sidebar.info("Kayıtlı yüz bulunamadı.")

    add_new_face_ui()

def display_unknown_face_item(face_id, face_data):
    try:
        img = Image.open(io.BytesIO(face_data))
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, width=80)
        with col2:
            new_name = st.text_input(f"İsim", key=f"unknown_{face_id}")
            if st.button("Ekle", key=f"add_{face_id}") and new_name:
                st.session_state.temp_faces[new_name.strip()] = face_data
                del st.session_state.unknown_faces[face_id]
                st.success(f"{new_name.strip()} eklendi!")
                st.rerun()
            if st.button("Sil", key=f"del_unknown_{face_id}"):
                del st.session_state.unknown_faces[face_id]
                st.rerun()
        st.markdown("---")
    except Exception as e:
        st.error(f"Görüntü yükleme hatası: {e}")

def unknown_faces_panel():
    with st.sidebar.expander("Tanınmayan Yüzler", expanded=True):
        if not st.session_state.unknown_faces:
            st.info("Tanınmayan yüz bulunamadı.")
            return
            
        for face_id, face_data in list(st.session_state.unknown_faces.items())[:10]:  
            display_unknown_face_item(face_id, face_data)

def camera_interface():
    st.header("📷 Live Camera Face Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        toggle_camera_button()
        process_camera_stream()
    
    with col2:
        st.subheader("Tanınmayan Yüzler")
        display_unknown_faces()

def toggle_camera_button():
    if st.button("Başlat" if not st.session_state.camera_running else "Durdur"):
        st.session_state.camera_running = not st.session_state.camera_running
        st.rerun()

def process_camera_stream():
    if st.session_state.camera_running:
        analyzer = handle_media_stream(0)
        if st.session_state.camera_running:
            analyzer.print_results()
        st.session_state.camera_running = False
        st.rerun()

def video_interface():
    st.header("🎞️ Video File Face Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Video seçin", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file:
            process_uploaded_video(uploaded_file)
    
    with col2:
        st.subheader("Tanınmayan Yüzler")
        display_unknown_faces()

def process_uploaded_video(uploaded_file):
    video_key = f"video_analysis_{uploaded_file.file_id=}"
    st.session_state.setdefault(video_key, {'running': False, 'path': None})

    handle_video_analysis_buttons(video_key, uploaded_file)
    run_video_analysis(video_key)

def handle_video_analysis_buttons(video_key, uploaded_file):
    if st.button("Analiz Başlat", key=f"start_{video_key}") and not st.session_state[video_key]['running']:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state[video_key]['path'] = tmp_file.name
        st.session_state[video_key]['running'] = True
        st.session_state['stop_video'] = False
        st.rerun()

    if st.session_state[video_key]['running']:
        if st.button("Analizi Durdur", key=f"stop_{video_key}"):
            st.session_state['stop_video'] = True

def run_video_analysis(video_key):
    if not st.session_state[video_key]['running']:
        return
        
    analyzer = handle_media_stream(st.session_state[video_key]['path'], is_camera=False)

    cleanup_video_file(video_key)
    
    if not st.session_state.get('stop_video', False):
        analyzer.print_results()

    reset_video_state(video_key)

def cleanup_video_file(video_key):
    if st.session_state[video_key]['path'] and os.path.exists(st.session_state[video_key]['path']):
        os.unlink(st.session_state[video_key]['path'])

def reset_video_state(video_key):
    st.session_state[video_key]['running'] = False
    st.session_state[video_key]['path'] = None
    st.session_state['stop_video'] = False
    st.rerun()

def save_unknown_face(face_id, name, face_data):
    st.session_state.temp_faces[name.strip()] = face_data
    del st.session_state.unknown_faces[face_id]
    st.success(f"{name.strip()} eklendi!")
    st.rerun()

def display_unknown_face_ui(face_id, face_data):
    st.image(io.BytesIO(face_data), width=150)
    name = st.text_input("İsim", key=f"name_{face_id}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Kaydet", key=f"save_{face_id}") and name:
            save_unknown_face(face_id, name, face_data)
    with col2:
        if st.button("Yoksay", key=f"ignore_{face_id}"):
            del st.session_state.unknown_faces[face_id]
            st.rerun()
    st.markdown("---")

def display_unknown_faces():
    if not st.session_state.unknown_faces:
        st.info("Tanınmayan yüz bulunamadı.")
        return
        
    for face_id, face_data in list(st.session_state.unknown_faces.items())[:5]:
        display_unknown_face_ui(face_id, face_data)

def main():
    st.set_page_config(layout="wide")
    st.title("🕵️ Face Analytics App")
    init_session()

    st.sidebar.title("Kontrol Paneli")
    mode = st.sidebar.radio(
        "Çalışma Modu",
        ("Kamera", "Video"),
        index=0
    )

    with st.sidebar.expander("⚙️ Ayarlar ve Yüz Yönetimi", expanded=True):
        settings_interface()
    
    if mode == "Kamera":
        camera_interface()
    else:
        video_interface()

if __name__ == "__main__":
    main()