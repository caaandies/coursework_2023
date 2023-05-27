import datetime
from functions import *
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    selected_option = option_menu("Main Menu", ["Home", "Forecasts", "Data"], 
        icons=["house", "cloud-drizzle", "clipboard-data"], menu_icon="cast")

if selected_option == "Home":
    st.title("Hello world!")
    st.markdown("This is a local web application that will allow you to use the program using a graphical interface")

elif selected_option == "Forecasts":
    st.subheader("Choose parameters")
    with st.container():
        col1, col2 = st.columns(2)

        selected_method = col1.radio(
            'Which method do you want to use?',
            ['Natural analogue', 'Calculated analogue']
        )

        selected_data = col2.radio(
            'On what data do you want to use the method?',
            ['daily mslp', 'monthly mslp']
        )
    st.subheader("Set the initial state")
    
    today = datetime.date.today()
    start_date = datetime.date(2000, 1, 1)
    end_date = today - datetime.timedelta(weeks=6)

    selected_date = st.date_input("Enter the date", start_date, min_value=start_date, max_value=end_date)
    st.write("Your date:", selected_date)
    if selected_data == "daily mslp":
        units = "days"
        file = "daily_mslp.nc"
    elif selected_data == "monthly mslp":
        units = "months"
        file = "monthly_mslp.nc"
    duration = st.number_input(f'Enter the forecast duration in {units}', step=1, min_value=1)

    st.markdown("You can upload a file with the initial state, otherwise the initial state will be considered the state at the time of the entered date")
    uploaded_file = st.file_uploader("Choose a file", type=['nc'])

    if st.button("Make forecast"):
        with st.spinner('Wait for it...'):
            forecast = Forecast(file, selected_date, duration, uploaded_file)
            if selected_method == "Natural analogue":
                init, prediction, analogue, rmse, corr, analogue_time = forecast.natural_analogue()
                rmse = np.round(rmse, decimals=2)
                corr = np.round(corr, decimals=4)
                st.pyplot(forecast.make_picture(init, "initial state"))
                st.pyplot(forecast.make_picture(analogue, f"natural analogue ({analogue_time}, rmse={str(rmse)}, corr={str(corr)})"))
                forecast.make_animation(prediction)
                st.image('forecast.gif')
            elif selected_method == "Calculated analogue":
                init, prediction, analog, rmse, corr = forecast.calculated_analogue()
                rmse = np.round(rmse, decimals=2)
                corr = np.round(corr, decimals=2)
                st.pyplot(forecast.make_picture(init, "initial state"))
                st.pyplot(forecast.make_picture(analog, f"calculated analogue (rmse={str(rmse)}, corr={str(corr)})"))
                forecast.make_animation(prediction)
                st.image('forecast.gif')
        
elif selected_option == "Data":
    st.subheader('Do you want to upload/update data?')
    if st.button('Upload/Update'):
        with st.spinner('Wait for it...'):
            update_data()
        st.success('Done!')

