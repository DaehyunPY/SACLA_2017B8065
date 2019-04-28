FROM fedora:29
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

# Install packages using dnf
ENV LANG=en_US.UTF-8
RUN dnf update -y \
    && dnf install -y \
        which bzip2 git clang texlive java-1.8.0-openjdk nodejs \
        https://downloads.sourceforge.net/project/mscorefonts2/rpms/msttcore-fonts-installer-2.6-1.noarch.rpm \
    && dnf clean all

# Install Anaconda
WORKDIR /root
ENV PATH /opt/anaconda/bin:$PATH
RUN curl -o anaconda.sh https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh \
    && bash anaconda.sh -bp /opt/anaconda \
    && rm anaconda.sh

# Build Pipenv project
WORKDIR /app
ENV PIPENV_VENV_IN_PROJECT=true \
    PATH=/app/.venv/bin:$PATH \
    PYSPARK_PYTHON=/app/.venv/bin/python
COPY Pipfile /app/
RUN pip install pipenv \
    && pipenv --python /opt/anaconda/bin/python --site-packages \
    && pipenv install --dev \
    && python -m jupyter labextension install \
        @jupyterlab/toc \
        @jupyter-widgets/jupyterlab-manager \
    && rm -fr /root/.cache

# Change permissions for multi-user
ENV HOME=/app \
    JAVA_TOOL_OPTIONS="-Duser.home=/app"
RUN chmod a+wx /app

EXPOSE 4040 8888
VOLUME [ "/app/.jupyter", "/app/.ivy2" ]
CMD [ "python" ]