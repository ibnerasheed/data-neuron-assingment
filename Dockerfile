FROM python:3.9.6
RUN mkdir app
COPY ./requirements.txt app
RUN pip install -r app/requirements.txt
COPY ./language_model_train.py app
COPY ./main.py app
COPY ./best_siamese_model.h5 app
COPY ./simple_text_model.py app
WORKDIR app
CMD ["uvicorn", "--host",  "0.0.0.0",  "--port",  "80",  "main:app"]

