FROM python:3.9-alpine as builder

WORKDIR /app


RUN pip install pipenv  && \
    apk add gcc g++ python3-dev libffi-dev

ENV PIPENV_VENV_IN_PROJECT=1

ADD Pipfile* /app/
RUN pipenv install --skip-lock


FROM python:3.9-alpine as prod

WORKDIR /app


ENV PIPENV_VENV_IN_PROJECT=1
RUN pip install pipenv

COPY --from=builder /app /app/
RUN ls -al && pipenv install

ADD app /app/app

EXPOSE 8000
CMD pipenv run python -m app
