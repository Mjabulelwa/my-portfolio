import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Doctor Booking System", layout="wide")

st.title(" Doctor Appointment Booking Dashboard")

# ================= CLASSES =================
class Doctor:
    def __init__(self, name, specialty):
        self.name = name
        self.specialty = specialty
        self.slots = []

class Patient:
    def __init__(self, name):
        self.name = name

class Booking:
    def __init__(self, doctor, patient, slot):
        self.doctor = doctor
        self.patient = patient
        self.slot = slot

    def confirm(self):
        return f"{self.patient.name} booked {self.doctor.name} at {self.slot}"

# ================= SESSION STORAGE =================
if "bookings" not in st.session_state:
    st.session_state.bookings = []

# ================= INPUT =================
st.sidebar.header(" Book Appointment")

doctor_name = st.sidebar.text_input("Doctor Name", "Dr Name")
specialty = st.sidebar.selectbox("Specialty", [
    "Dentist", "Cardiologist", "General",
    "Dermatologist", "Gynecologist", "Psychiatrist"
])

slots = ["09:00 AM", "10:00 AM", "11:00 AM", "02:00 PM", "03:00 PM"]

selected_slot = st.sidebar.selectbox("Select Time Slot", slots)
patient_name = st.sidebar.text_input("Patient Name")

if st.sidebar.button("Book Appointment"):
    if patient_name:
        booking = Booking(Doctor(doctor_name, specialty), Patient(patient_name), selected_slot)

        st.session_state.bookings.append({
            "Doctor": doctor_name,
            "Specialty": specialty,
            "Patient": patient_name,
            "Slot": selected_slot
        })

        st.success(booking.confirm())
    else:
        st.error("Please enter patient name")

# ================= DISPLAY =================
st.subheader("Booking Records")

if st.session_state.bookings:
    df = pd.DataFrame(st.session_state.bookings)
    st.dataframe(df)

    # ================= KPI =================
    col1, col2 = st.columns(2)
    col1.metric("Total Bookings", len(df))
    col2.metric("Unique Patients", df["Patient"].nunique())

    # ================= PLOTLY =================
    st.subheader("Interactive Insights (Plotly)")

    col1, col2 = st.columns(2)

    fig1 = px.bar(df, x="Doctor", title="Bookings per Doctor", color="Doctor")
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(df, names="Slot", title="Bookings by Time Slot")
    col2.plotly_chart(fig2, use_container_width=True)

    # ================= SEABORN VISUALS =================
    st.subheader("Advanced Insights (Seaborn)")

    col3, col4 = st.columns(2)

    # 1. Countplot (Specialty distribution)
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x="Specialty", palette="viridis", ax=ax3)
    ax3.set_title("Specialty Distribution")
    ax3.tick_params(axis='x', rotation=30)
    col3.pyplot(fig3)

    # 2. Countplot (Slots)
    fig4, ax4 = plt.subplots()
    sns.countplot(data=df, x="Slot", palette="coolwarm", ax=ax4)
    ax4.set_title("Appointments per Time Slot")
    col4.pyplot(fig4)

    # ================= HEATMAP =================
    st.subheader("Booking Heatmap (Doctor vs Slot)")

    pivot = df.pivot_table(index="Doctor", columns="Slot", aggfunc="size", fill_value=0)

    fig5, ax5 = plt.subplots()
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax5)
    st.pyplot(fig5)

    # ================= BOX PLOT =================
    st.subheader("Booking Spread Analysis")

    df["BookingCount"] = 1

    fig6, ax6 = plt.subplots()
    sns.boxplot(data=df, x="Specialty", y="BookingCount", palette="Set2", ax=ax6)
    ax6.set_title("Booking Variation by Specialty")
    ax6.tick_params(axis='x', rotation=30)
    st.pyplot(fig6)

else:
    st.info("No bookings yet. Use the sidebar to add appointments.")