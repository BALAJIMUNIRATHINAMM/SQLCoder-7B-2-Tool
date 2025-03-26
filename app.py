import os
print(os.listdir("/"))
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import asyncio

# Load the model and tokenizer
MODEL_NAME = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)

def generate_sql_query(nl_query, schema, temperature, num_beams):
    """Generates SQL query from natural language input."""
    prompt = f"""
    ### Task
    Generate a SQL query to answer [QUESTION]{nl_query}[/QUESTION]

    ### Database Schema
    The query will run on a database with the following schema:
    {schema}

    ### Answer
    Given the database schema, here is the SQL query that [QUESTION]{nl_query}[/QUESTION]
    [SQL]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=temperature, num_beams=num_beams)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query.strip()

# Fix for RuntimeError: This event loop is already running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))

# Streamlit UI
st.set_page_config(page_title="SQLCoder-7B-2 App", layout="wide")
st.title("SQLCoder-7B-2: Natural Language to SQL Converter")

st.sidebar.header("Settings")
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
num_beams = st.sidebar.slider("Number of Beams", 1, 10, 5)

nl_query = st.text_area("Enter your natural language query:")
schema = st.text_area("Enter the database schema (DDL statements):")

if st.button("Generate SQL Query"):
    if nl_query and schema:
        sql_query = generate_sql_query(nl_query, schema, temperature, num_beams)
        st.subheader("Generated SQL Query:")
        st.code(sql_query, language='sql')
    else:
        st.warning("Please provide both a query and a database schema.")

st.sidebar.markdown("**🔹 Powered by SQLCoder-7B-2**")
