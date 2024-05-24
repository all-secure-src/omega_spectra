
# Use the official Python image as a parent image
FROM python:3.10.14

# Set the working directory
WORKDIR /code
#
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#
COPY ./app /code/app

# Expose the port FastAPI will run on
EXPOSE 8080

# Define the command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080",  "--workers", "20"]