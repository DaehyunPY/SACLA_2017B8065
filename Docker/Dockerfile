FROM fedora:29
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

ENV LANG en_US.UTF-8

WORKDIR /root
RUN dnf update -y \
    && dnf install -y \
        bzip2 \
        git \
        java-1.8.0-openjdk \
        texlive \
        which \
    && dnf clean all

RUN curl -o /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.3.30-Linux-x86_64.sh \
    && bash /root/miniconda.sh -bp /opt/anaconda \
    && rm /root/miniconda.sh
ENV PATH /opt/anaconda/bin:$PATH

COPY conda-env.yaml /root/
RUN conda install anaconda \
    && conda env update -n root -f /root/conda-env.yaml \
    && conda clean -tipsy \
    && rm -fr \
        /root/conda-env.yaml \
        /root/.cache

WORKDIR /app
RUN chmod a+wx /app
ENV HOME /app
ENV JAVA_TOOL_OPTIONS -Duser.home=/app
CMD [ "python" ]
