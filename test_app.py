import streamlit as st

st.set_page_config(page_title="Minimal Test App", layout="centered")

st.title("Hello from Render!")
st.write("If you can see this without a 502, your deployment is fine.")
st.button("Click me")