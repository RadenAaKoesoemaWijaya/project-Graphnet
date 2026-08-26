import streamlit as st
from system_status import show_system_status_page

def show_status_page():
    st.info("Status Sistem menampilkan kesehatan aplikasi, perangkat komputasi, model, cache, dan operasi terakhir untuk membantu diagnosis sebelum menjalankan proses besar.")
    show_system_status_page()

