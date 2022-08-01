FROM 10jqkaaicubes/cuda:11.0-py3.8.5

COPY ./ /home/jovyan/causality_extraction_demo

RUN cd /home/jovyan/causality_extraction_demo  && \
    python -m pip install -r requirements.txt 