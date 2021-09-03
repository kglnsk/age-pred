FROM continuumio/anaconda3:latest
WORKDIR /usr/src/app
COPY . .

RUN conda config --add channels conda-forge
RUN conda install -y catboost
RUN conda install -y xgboost 

CMD ["solve.py"]
ENTRYPOINT ["python3"]