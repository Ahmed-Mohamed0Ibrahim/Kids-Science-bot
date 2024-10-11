# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the current directory contents into the container at /usr/src/app
COPY ./requirements.txt /code/requirements.txt

# Install any necessary libraries specified in requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# add python code
COPY ./app/ /code/app/

# Run your script
CMD ["streamlit", "run","app/app.py"]
