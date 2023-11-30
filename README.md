# Mozart Mistral Project

## Description
The Mozart Mistral Project is a Python-based application designed to interact with a language model using the `modal` library. Its primary function is to generate text based on the input provided, leveraging various Python libraries and a unique approach to text generation.

## Table of Contents
- [Mozart Mistral Project](#mozart-mistral-project)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Code Examples](#code-examples)
  - [Running the Application](#running-the-application)

## Dependencies
This project relies on several Python libraries including `modal`, `torch`, `pydantic`, `sentence-transformers`, among others. Refer to the `pyproject.toml` file for specific version requirements.

## Installation
To install the Mozart Mistral Project, follow these steps:
1. Clone the repository:
2. Install the required dependencies:

## Usage
The main functionality of the project revolves around generating text based on input. This is handled through a FastAPI application, with endpoints for generating embeddings and interacting with chat models.

## Code Examples
The main application is contained in `app.py`, which includes the FastAPI application logic and the `MozartMistral` class for text generation.

Chat models are defined in `chat_models.py` and `mozart_mistral.py`, and are used for generating text based on input messages.

## Running the Application
To run the application, execute the `local_main` function in `app.py`. This sends a request to the main endpoint with a sample text and prints the result.

```python
@stub.local_entrypoint()
def local_main():
 request_data = {"texts": ["Sample text"]}
 print(main(request_data))

if __name__ == "__main__":
 local_main()

