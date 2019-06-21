FROM daehyunpy/sacla-2017b8065:20190524
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

ARG NB_USER=user
ARG NB_UID=1000
ARG HOME=/app
USER root
RUN adduser --uid ${NB_UID} --home-dir ${HOME} ${NB_USER}
COPY . ${HOME}
RUN chown -R ${NB_UID} ${HOME}

WORKDIR /app
USER ${NB_USER}
CMD [ "python" ]
