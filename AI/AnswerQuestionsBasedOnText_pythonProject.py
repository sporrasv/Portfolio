# Creating an AI-powered question-answering program that works offline requires leveraging local models. We can use transformers from the Hugging Face library to load and use pre-trained language models (like DistilBERT) for answering questions based on user-provided text. This example will use a lightweight language model (DistilBERT) to fit offline use better and avoid high memory usage.

# Explanation
# Load Model: We use the distilbert-base-uncased-distilled-squad model, which is lightweight and pre-trained on question-answering tasks.
# User Input: The program prompts the user to input a reference text.
# Question-Answering Loop: The user can ask questions based on the input text, and the model will return answers. Type exit to end the program.

# Notes
# Offline Use: The first time you run it, the model will be downloaded, so ensure you run it once with an internet connection. Afterward, it should work offline as the model is cached locally.
# Limitations: This example works best with smaller, straightforward texts since offline models are typically less robust than larger, online models.
# This setup provides a simple, locally-run question-answering system using pre-trained models for offline use!

# STEPS:
# Step 1 First, make sure you have the necessary packages installed. Run the following bash code:
# pip install transformers torch

# Step 2 Run the program: python your_program.py



from transformers import pipeline

# Load the model and set up the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")


def get_text_input():
    print("Enter the text you'd like to use as a reference:")
    return input("\n> ")


def ask_questions(context):
    print("\nYou can now ask questions based on the text. Type 'exit' to quit.\n")
    while True:
        question = input("Your Question: ")

        if question.lower() == 'exit':
            print("Exiting the program.")
            break

        # Use the QA pipeline to answer the question
        result = qa_pipeline(question=question, context=context)
        print(f"Answer: {result['answer']}\n")


if __name__ == "__main__":
    text_context = get_text_input()
    ask_questions(text_context)
