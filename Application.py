import streamlit as st
import openai
import boto3
import json
import pandas as pd
from io import BytesIO

# OpenAI API Key
openai.api_key = 'sk-ZQuwt-NjRD5O6VVERM6Pittqk-VsICiOUploWQC50-T3BlbkFJ_gq7oVX_Xt5psrSHloWeLS-jEJzGw7-glGdDoBX_0A'

# AWS S3 Configuration
s3_client = boto3.client('s3')
s3_bucket = 'gaiaproject'

# Load metadata.jsonl from S3
def load_metadata():
    metadata_key = 'gaia/2023/validation/metadata.jsonl'  # Path in your S3 bucket
    response = s3_client.get_object(Bucket=s3_bucket, Key=metadata_key)
    content = response['Body'].read().decode('utf-8')
    metadata_lines = content.splitlines()
    metadata = [json.loads(line) for line in metadata_lines]
    return pd.DataFrame(metadata)

# Send a prompt to the OpenAI model
def query_openai_model(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=prompt, 
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Compare OpenAI's answer to the final answer in the metadata
def compare_answers(openai_answer, final_answer):
    return openai_answer.lower() == final_answer.lower()

# Main Streamlit app
def main():
    st.title("GAIA Dataset Model Evaluation Tool")

    # Load and display metadata from S3
    st.header("Validation Test Case Selection")
    metadata_df = load_metadata()
    test_case_id = st.selectbox("Select a Test Case", metadata_df["task_id"].unique())

    selected_test_case = metadata_df[metadata_df["task_id"] == test_case_id].iloc[0]
    st.write("Question:", selected_test_case["Question"])

    # Query OpenAI with the selected test case
    if st.button("Ask OpenAI"):
        openai_answer = query_openai_model(selected_test_case["Question"], selected_test_case.get("Annotator Metadata", {}).get("Steps", ""))
        st.write("OpenAI Answer:", openai_answer)

        # Compare OpenAI answer with final answer
        final_answer = selected_test_case["Final answer"]
        correct = compare_answers(openai_answer, final_answer)
        st.write(f"Is OpenAI's answer correct? {'Yes' if correct else 'No'}")

        # Option to modify Annotator steps
        if not correct:
            st.write("Modify the Annotator Steps to improve the model:")
            modified_steps = st.text_area("Annotator Steps", selected_test_case["Annotator Metadata"]["Steps"])
            if st.button("Re-evaluate"):
                revised_openai_answer = query_openai_model(selected_test_case["Question"], modified_steps)
                st.write("Revised OpenAI Answer:", revised_openai_answer)
                if compare_answers(revised_openai_answer, final_answer):
                    st.write("The revised OpenAI answer is correct.")
                else:
                    st.write("The revised OpenAI answer is still incorrect.")

    # Feedback and Reports
    st.header("User Feedback")
    feedback = st.text_area("Provide your feedback on this evaluation")
    if st.button("Submit Feedback"):
        st.write("Thank you for your feedback!")

    # Visualization (Dummy visualization for now)
    st.header("Reports and Visualization")
    st.write("This section will contain visualizations based on user feedback and model evaluation results.")

if __name__ == "__main__":
    main()
