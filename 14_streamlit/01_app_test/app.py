import streamlit as st

# Adding title of your app
st.title('My first app in Streamlit ')

# Adding content to your app
st.write('Hello, world!')

# user input
num = st.slider('Your Age', 0, 100, 18)

# display the number
st.write('Your age is: ', num)

# adding button 
if st.button('Say Hello'):
    st.write('Hello')
else:
    st.write('Goodbye')

# Add radio buttons
gender = st.radio('What is your gender?', ('Male', 'Female', 'Other'))
st.write('You selected:', gender)

# Add a selectbox
region = st.selectbox('Region', ('Pakistan', 'India', 'Nepal', 'Bhutan', 'Sri Lanka', 'Iran', 'China'))
st.write('You selected:', region)

# Add a selectbox at the sidebar
device = st.sidebar.selectbox('How would you like to be contacted?', ('Email', 'Home phone', 'Mobile phone'))
st.write('You selected:', device)

# Add a text input at the sidebar
num = st.sidebar.text_input('Enter Your number')
st.write('You entered:', num)