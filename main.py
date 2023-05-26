import datetime
from functions import *
import streamlit as st
from streamlit_option_menu import option_menu
from io import StringIO

time_units_dict = {
              "daily mslp": "days",
              "monthly mslp": "months"
              }

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "Forecasts", "Data"], 
        icons=["house", "cloud-drizzle", "clipboard-data"], menu_icon="cast")

if selected == "Home":
    st.title("Hello world!")
    st.markdown("This is a local web application that will allow you to use the program using a graphical interface")

elif selected == "Forecasts":
    st.subheader("Choose parameters")
    with st.container():
        col1, col2 = st.columns(2)

        method = col1.radio(
            'Which method do you want to use?',
            ['Natural analogue', 'Calculated analogue']
        )

        data = col2.radio(
            'On what data do you want to use the method?',
            ['daily mslp', 'monthly mslp']
        )
    st.subheader("Set the initial state")

    

    
    today = datetime.date.today()
    start_date = datetime.date(2000, 1, 1)
    end_date = today - datetime.timedelta(weeks=6)

    selected_date = st.date_input("Enter the date", start_date, min_value=start_date, max_value=end_date)
    st.write("Your date:", selected_date)

    duration = st.number_input(f'enter the forecast duration in {time_units_dict[data]}', step=1, min_value=1)

    if st.button("Make forecast"):
        with st.spinner('Wait for it...'):
            if method == "Natural analogue":
                init, forecast, analog, rmse, corr, analog_time = natural_analogue(selected_date, data, duration)
                st.pyplot(make_picture(init), "initial state")
                st.pyplot(make_picture(analog), f"natural analog from {analog_time} with rmse={rmse} correlation={corr}")
                make_animation(forecast, selected_date, data)
                st.image('forecast.gif')
            else:
                init, forecast, analog, rmse, corr = calculated_analogue(selected_date, data, duration)
                st.pyplot(make_picture(init), "initial state")
                st.pyplot(make_picture(analog), "calculated analogue with rmse={rmse} correlation={corr}")
                make_animation(forecast, selected_date, data)
                st.image('forecast.gif')
        
elif selected == "Data":
    st.subheader('Do you want to upload/update data?')
    if st.button('Upload/Update'):
        with st.spinner('Wait for it...'):
            update_data()
        st.success('Done!')

