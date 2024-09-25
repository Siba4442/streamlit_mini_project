import streamlit as st
import candidate
import recruiter
import ATS

# Sidebar for navigation using a radio button
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Candidate", "Recruiter","ATS Score"))

# Home page content
if page == "Home":
    st.title("Welcome to the Home Page")
    st.write("Use the sidebar to navigate to the Candidate or Recruiter pages.")

# Candidate page
elif page == "Candidate":
    candidate.app()

# Recruiter page
elif page == "Recruiter":
    recruiter.app()

#ATS Score page
elif page == "ATS Score":
    ATS.app()