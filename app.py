import streamlit as st
import cv2
import tempfile
from Analyzer import FaceAnalyzer
import os
import io
from PIL import Image

def init_session():
    st.session_state.setdefault('camera_running', False)
    st.session_state.setdefault('model_path')
    st.session_state.setdefault('face_match_threshold', 0.6)
    st.session_state.setdefault('max_faces', 5)
    st.session_state.setdefault('frame_skip', 2)
    st.session_state.setdefault('temp_faces', {})
    st.session_state.setdefault('stop_video', False)
    st.session_state.setdefault('speaking_times', {})

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
    
    # Konuşma sürelerini ekle
    y_offset = 30
    for i, (name, duration) in enumerate(analyzer.speaking_times.items()):
        cv2.putText(processed, f"{name}: {duration:.1f}s", 
                   (10, 30 + y_offset*i), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    # Store speaking times in session state for later use
    st.session_state.speaking_times = analyzer.speaking_times
    
    frame_placeholder.image(
        cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
        channels="RGB"
    )

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
            
            if st.session_state.get('stop_video', False):
                break
            
    finally:
        cap.release()
        if not is_camera:
            os.unlink(input_source) if os.path.exists(input_source) else None
    return analyzer

def display_model_settings():
    st.sidebar.header("⚙️ FaceAnalyzer Ayarları")
    
    try:
        model_files = [f"src/{f}" for f in os.listdir('src') if f.endswith('.pt')]
        st.session_state.model_path = st.sidebar.radio(
            "Model Seçimi",
            model_files,
            index=0,
            format_func=lambda x: x.split("/")[-1].split(".")[0]
        )
    except FileNotFoundError:
        st.sidebar.error("src klasörü bulunamadı!")
    
    st.sidebar.slider("Yüz Eşleşme Eşiği", 0.3, 1.0, value=st.session_state.face_match_threshold, key="face_match_threshold")
    st.sidebar.slider("Maksimum Yüz Sayısı", 1, 10, value=st.session_state.max_faces, key="max_faces")
    st.sidebar.slider("Frame Atlatma", 1, 10, value=st.session_state.frame_skip, key="frame_skip")

def display_saved_face(face_name, in_sidebar=True):
    if in_sidebar:
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.image(io.BytesIO(st.session_state.temp_faces[face_name]), width=60)
    with col2:
        st.write(face_name)
        speaking_time = st.session_state.speaking_times.get(face_name, 0)
        st.caption(f"{speaking_time:.1f} saniye")
    with col3:
        if st.button("❌", key=f"del_{face_name}{'_panel' if not in_sidebar else ''}"):
            del st.session_state.temp_faces[face_name]
            st.rerun()

def display_temp_faces_panel():
    st.markdown("---")
    st.subheader("Geçici Yüzler")

    if st.session_state.temp_faces:
        for face_name in list(st.session_state.temp_faces.keys()):
            display_saved_face(face_name, in_sidebar=False)
    else:
        st.info("Kayıtlı yüz bulunamadı.")

def add_new_face_ui():
    st.markdown("**Yeni Yüz Ekle**")
    uploaded_face = st.file_uploader("Yüz fotoğrafı yükle", type=["jpg", "jpeg", "png"])
    new_name = st.text_input("Yüz ismi")
    
    if st.button("Yüz Ekle") and uploaded_face and new_name:
        try:
            st.session_state.temp_faces[new_name.strip()] = uploaded_face.getvalue()
            st.success(f"{new_name.strip()} eklendi!")
            st.rerun()
        except Exception as e:
            st.error(f"Hata: {e}")

def settings_interface():
    with st.sidebar.expander("⚙️ Ayarlar", expanded=True):
        display_model_settings()

def display_speech_results(analyzer):
    st.subheader("🎙️ Konuşma Süreleri")
    if not analyzer.speaking_times:
        st.info("Konuşma tespit edilmedi")
        return
    
    for name, duration in analyzer.speaking_times.items():
        st.metric(label=name, value=f"{duration:.1f} saniye")

def camera_interface():
    st.header("📷 Canlı Kamera Analizi")
    
    left_col, right_col = st.columns([3, 1])
    
    with left_col:
        if st.button("Başlat" if not st.session_state.camera_running else "Durdur"):
            st.session_state.camera_running = not st.session_state.camera_running
            st.rerun()

        if st.session_state.camera_running:
            analyzer = handle_media_stream(0)
            display_speech_results(analyzer)
            st.session_state.camera_running = False
            st.rerun()
    
    with right_col:
        add_new_face_ui()
        display_temp_faces_panel()

def video_interface():
    st.header("🎞️ Video Dosyası Analizi")
    
    left_col, right_col = st.columns([3, 1])
    
    with left_col:
        uploaded_file = st.file_uploader("Video yükle (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            st.video(uploaded_file.getvalue())
            
            if st.button("Analiz Başlat"):
                analyzer = handle_media_stream(video_path, is_camera=False)
                display_speech_results(analyzer)
                os.unlink(video_path)
    
    with right_col:
        add_new_face_ui()
        display_temp_faces_panel()

def main():
    st.set_page_config(page_title="Face Analytics", layout="wide")
    st.title("🕵️ Yüz Analiz Uygulaması")
    init_session()
    
    mode = st.sidebar.radio(
        "Çalışma Modu",
        ("📷 Kamera", "🎞️ Video"),
        index=0
    )
    
    settings_interface()
    
    if "Kamera" in mode:
        camera_interface()
    else:
        video_interface()

if __name__ == "__main__":
    main()