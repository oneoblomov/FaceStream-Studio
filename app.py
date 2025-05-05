import streamlit as st
import cv2
import tempfile
from Analyzer import FaceAnalyzer
import os
import io
import pandas as pd
from datetime import datetime
import json

class FaceStreamStudioApp:
    def __init__(self):
        self.div = "</div>"
        self.LANGUAGES = self.load_languages()
        self.DEFAULT_LANGUAGE = next(iter(self.LANGUAGES), "English")
        self.init_session()

    # --- Language & Session Management ---
    def load_languages(self):
        try:
            with open("languages.json", "r", encoding="utf-8") as f:
                langs = json.load(f)
                if not isinstance(langs, dict):
                    raise ValueError("languages.json format is invalid.")
                return langs
        except Exception as e:
            st.error(f"Dil dosyası yüklenemedi: {e}")
            return {}

    def get_lang(self):
        return self.LANGUAGES.get(st.session_state.get("language", self.DEFAULT_LANGUAGE), self.LANGUAGES.get(self.DEFAULT_LANGUAGE, {}))

    def init_session(self):
        if "language" not in st.session_state:
            st.session_state["language"] = self.DEFAULT_LANGUAGE
        defaults = {
            'camera_running': False,
            'model_path': None,
            'face_match_threshold': 0.6,
            'max_faces': 5,
            'frame_skip': 2,
            'temp_faces': {},
            'stop_video': False,
            'speaking_times': {},
            'speaking_threshold': 0.03,
            'show_names': True,
            'show_times': True,
            'show_emotion': True,
            'theme': 'Light',
            'show_bounding_boxes': True
        }
        for k, v in defaults.items():
            st.session_state.setdefault(k, v)

    # --- Analyzer Helpers ---
    def create_analyzer(self):
        return FaceAnalyzer(
            model_path=st.session_state.model_path,
            temp_faces=st.session_state.temp_faces
        )

    def configure_analyzer(self, analyzer):
        analyzer.FACE_MATCH_THRESHOLD = st.session_state.face_match_threshold
        analyzer.MAX_FACES = st.session_state.max_faces
        analyzer.FRAME_SKIP = st.session_state.frame_skip
        analyzer.SPEAKING_LIP_DIST_THRESHOLD = st.session_state.speaking_threshold
        analyzer.show_names = st.session_state.show_names
        analyzer.show_times = st.session_state.show_times
        analyzer.show_emotion = st.session_state.show_emotion
        analyzer.show_bounding_boxes = st.session_state.show_bounding_boxes
        return analyzer

    def process_frame(self, analyzer, frame, frame_placeholder):
        LANG = self.get_lang()
        processed = analyzer.process_frame(frame)
        if analyzer.show_times:
            y_offset = 30
            for i, (name, duration) in enumerate(analyzer.speaking_times.items()):
                cv2.putText(processed, LANG["name_duration"].format(name, duration),
                            (10, 30 + y_offset*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        st.session_state.speaking_times = analyzer.speaking_times
        frame_placeholder.image(
            cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

    def handle_media_stream(self, input_source, is_camera=True):
        analyzer = self.configure_analyzer(self.create_analyzer())
        frame_placeholder = st.empty()
        cap = cv2.VideoCapture(input_source if is_camera else str(input_source))
        LANG = self.get_lang()
        if not cap.isOpened():
            st.error(LANG["video_error"])
            return analyzer
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                self.process_frame(analyzer, frame, frame_placeholder)
                if st.session_state.get('stop_video', False):
                    break
        finally:
            cap.release()
            if not is_camera and os.path.exists(input_source):
                os.unlink(input_source)
        return analyzer

    # --- UI: Settings ---
    def model_settings(self):
        LANG = self.get_lang()
        with st.sidebar.expander(LANG["model_settings"], expanded=False):
            try:
                model_files = [f"src/{f}" for f in os.listdir('src') if f.endswith('.pt')]
                st.session_state.model_path = st.radio(
                    LANG["model_settings"],
                    model_files,
                    index=0,
                    format_func=lambda x: x.split("/")[-1].split(".")[0]
                )
            except FileNotFoundError:
                st.error(LANG["src_not_found"])
            st.slider(LANG["face_match_threshold"], 0.3, 1.0, value=st.session_state.face_match_threshold, key="face_match_threshold")
            st.slider(LANG["max_faces"], 1, 10, value=st.session_state.max_faces, key="max_faces")
            st.slider(LANG["frame_skip"], 1, 10, value=st.session_state.frame_skip, key="frame_skip")

    def detection_settings(self):
        LANG = self.get_lang()
        with st.sidebar.expander(LANG["detection_settings"], expanded=False):
            st.slider(
                LANG["speaking_threshold"], 
                0.01, 0.1, 
                value=st.session_state.speaking_threshold,
                key="speaking_threshold", 
                format="%.3f", 
                help=LANG.get("speaking_threshold_help", "Lower value = more sensitive detection")
            )

    def display_settings(self):
        LANG = self.get_lang()
        with st.sidebar.expander(LANG["display_settings"], expanded=False):
            st.checkbox(LANG["show_names"], value=st.session_state.show_names, key="show_names")
            st.checkbox(LANG["show_times"], value=st.session_state.show_times, key="show_times")
            st.checkbox(LANG["show_emotion"], value=st.session_state.show_emotion, key="show_emotion")
            st.checkbox(LANG["show_bounding_boxes"], value=st.session_state.show_bounding_boxes, key="show_bounding_boxes")

    def ui_settings(self):
        LANG = self.get_lang()
        with st.sidebar.expander(LANG["ui_settings"], expanded=False):
            st.selectbox(
                LANG["language"],
                options=list(self.LANGUAGES.keys()),
                key="language",
                index=list(self.LANGUAGES.keys()).index(st.session_state.language) if st.session_state.language in self.LANGUAGES else 0,
            )

    def export_results(self):
        LANG = self.get_lang()
        results = []
        column_name = LANG.get("name_column", "Name")
        for name, duration in st.session_state.speaking_times.items():
            results.append({
                column_name: name,
                LANG["speech_times"]: duration
            })
        if results:
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False).encode('utf-8')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label=LANG["download_csv_btn"],
                data=csv,
                file_name=f"{LANG.get('csv_filename', 'results')}_{timestamp}.csv",
                mime="text/csv"
            )

    def settings_interface(self):
        LANG = self.get_lang()
        st.sidebar.header(LANG["settings"])
        self.model_settings()
        self.detection_settings()
        self.display_settings()
        self.ui_settings()
        with st.sidebar.expander(LANG["results_export"], expanded=False):
            if st.button(LANG["download_csv"]) and st.session_state.speaking_times:
                self.export_results()

    # --- UI: Faces ---
    def display_saved_face(self, face_name, in_sidebar=True):
        LANG = self.get_lang()
        cols = st.sidebar.columns([1, 2, 1]) if in_sidebar else st.columns([1, 2, 1])
        with cols[0]:
            st.image(io.BytesIO(st.session_state.temp_faces[face_name]), width=60)
        with cols[1]:
            st.write(face_name)
            speaking_time = st.session_state.speaking_times.get(face_name, 0)
            st.caption(f"{speaking_time:.1f} {LANG['seconds']}")
        with cols[2]:
            if st.button(LANG["delete_face"], key=f"del_{face_name}{'_panel' if not in_sidebar else ''}"):
                del st.session_state.temp_faces[face_name]
                st.session_state.speaking_times.pop(face_name, None)
                st.rerun()

    def display_temp_faces_panel(self):
        LANG = self.get_lang()
        st.markdown("""<div class="card">""", unsafe_allow_html=True)
        if st.session_state.temp_faces:
            for face_name in list(st.session_state.temp_faces.keys()):
                self.display_saved_face(face_name, in_sidebar=False)
        else:
            st.info(LANG["no_faces"])
        st.markdown(self.div, unsafe_allow_html=True)

    def add_new_face_ui(self):
        LANG = self.get_lang()
        st.markdown(f"""
        <div class="card">
        <h3>{LANG['add_face']}</h3>
        """, unsafe_allow_html=True)
        uploaded_face = st.file_uploader(LANG["upload_face"], type=["jpg", "jpeg", "png"])
        selected_filename = uploaded_face.name if uploaded_face else ""
        new_name = st.text_input(LANG["face_name"], value=selected_filename.split('.')[0] if selected_filename else "")
        if st.button(LANG["add_face_btn"], type="primary") and uploaded_face and new_name:
            try:
                st.session_state.temp_faces[new_name.strip()] = uploaded_face.getvalue()
                st.success(LANG["face_added"].format(new_name.strip()))
                st.rerun()
            except Exception as e:
                st.error(LANG["face_error"].format(e))
        st.markdown(self.div, unsafe_allow_html=True)

    # --- UI: Results ---
    def display_speech_results(self, analyzer):
        LANG = self.get_lang()
        st.markdown(f"""
        <div class="card results-card">
        <h3>{LANG['speech_times']}</h3>
        """, unsafe_allow_html=True)
        if not analyzer.speaking_times:
            st.info(LANG["no_speech"])
        else:
            cols = st.columns(min(3, len(analyzer.speaking_times)))
            for i, (name, duration) in enumerate(analyzer.speaking_times.items()):
                with cols[i % len(cols)]:
                    st.metric(label=name, value=f"{duration:.1f} {LANG['seconds']}")
        st.markdown(self.div, unsafe_allow_html=True)

    # --- Main Interfaces ---
    def camera_interface(self):
        LANG = self.get_lang()
        st.markdown(f"""
        <div class="card">
        <h2>{LANG['camera_analysis']}</h2>
        """, unsafe_allow_html=True)
        left_col, right_col = st.columns([3, 1])
        with left_col:
            if st.button("📹 " + (LANG["stop"] if st.session_state.camera_running else LANG["start"]),
                        type="primary" if not st.session_state.camera_running else "secondary"):
                st.session_state.camera_running = not st.session_state.camera_running
                st.rerun()
            placeholder = st.empty()
            if st.session_state.camera_running:
                with placeholder.container():
                    st.markdown('<div style="border-radius:10px; overflow:hidden;">', unsafe_allow_html=True)
                    analyzer = self.handle_media_stream(0)
                    st.markdown('</div>', unsafe_allow_html=True)
                    self.display_speech_results(analyzer)
                st.session_state.camera_running = False
                st.rerun()
        with right_col:
            self.display_temp_faces_panel()
            self.add_new_face_ui()
        st.markdown(self.div, unsafe_allow_html=True)

    def video_interface(self):
        LANG = self.get_lang()
        st.markdown(f"""
        <div class="card">
        <h2>{LANG['video_analysis']}</h2>
        """, unsafe_allow_html=True)
        left_col, right_col = st.columns([3, 1])
        with left_col:
            uploaded_file = st.file_uploader(LANG["upload_video"], type=["mp4", "avi", "mov"])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
                if st.button(LANG["analyze_video"], type="primary"):
                    with st.spinner(LANG["analyzing_video"]):
                        analyzer = self.handle_media_stream(video_path, is_camera=False)
                        self.display_speech_results(analyzer)
        with right_col:
            self.display_temp_faces_panel()
            self.add_new_face_ui()
        st.markdown(self.div, unsafe_allow_html=True)

    def main(self):
        LANG = self.get_lang()
        st.set_page_config(page_title="FaceStream Studio", layout="wide")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1 style="display:inline;">{LANG.get('app_title', 'FaceStream Studio')}</h1>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="text-align: left; margin-bottom: 2rem;">
                <h3 style="display:inline;">{LANG.get('face_list', 'Saved Faces')}</h3>
            </div>
            """, unsafe_allow_html=True)
        mode_options = [LANG.get("mode_camera", "Camera"), LANG.get("mode_video", "Video")]
        mode = st.sidebar.radio(
            LANG.get("mode", "Mode"),
            mode_options,
            index=0,
            format_func=lambda x: x
        )
        self.settings_interface()
        if mode_options[0] in mode:
            self.camera_interface()
        else:
            self.video_interface()

if __name__ == "__main__":
    app = FaceStreamStudioApp()
    app.main()
