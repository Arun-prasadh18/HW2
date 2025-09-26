import streamlit as st
from openai import OpenAI

# Show title and description.
# st.title("📄 Document question answering")
st.set_page_config(page_title=None, page_icon=None, layout="centered",
initial_sidebar_state="auto", menu_items=None)
first_page=st.Page("HWs/HW1.py", title="First page", icon=None,
url_path=None, default=False)
second_page=st.Page("HWs/HW2.py", title="Second page", icon=None,
url_path=None, default=False)
Third_page=st.Page("HWs/HW3.py", title="Third page", icon=None,
url_path=None, default=False)
Lab4=st.Page("HWs/HW4.py", title="Fourth page", icon=None,
url_path=None, default=False)
Lab5=st.Page("lab5.py", title="Fifth page", icon=None,
url_path=None, default=False)
LAB5b=st.Page("LAB5.py", title="Sixth page", icon=None,
url_path=None, default=False)
HW5=st.Page("HWs/HW5.py", title="HW5", icon=None,
url_path=None, default=True)
pg=st.navigation([first_page,second_page,Third_page,Lab4,Lab5,LAB5b,HW5])
# st.set_page_config(page_title="Data Manager",page_icon=":material/edit:")
# with st.sidebar:
#     add_radio=st.radio("choose a option",("Summarize the document in 100 words", 
#                                           "Summarize the document in 2 connecting paragraphs",
#  "Summarize the document in 5 bullet points"))
#     model_radio=st.radio("choose a option",("GPT-MINI","GPT-NANO"))
#     st.checkbox("Use advanced model", value=False)
#     # if st.checkbox("Use advanced model"):
#     #     model_advanced_radio=st.radio("choose a option",("GPT-MINI"))
pg.run()