FROM python:3.10-bullseye AS pylucene-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    ant \
    build-essential \
    python3-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV JCC_JDK=${JAVA_HOME}
ENV PATH="${JAVA_HOME}/bin:${PATH}"

RUN JH="$JAVA_HOME" && \
    mkdir -p "$JH/jre/lib/amd64/server" && \
    if [ -f "$JH/lib/server/libjvm.so" ]; then \
        ln -sf "$JH/lib/server/libjvm.so" "$JH/jre/lib/amd64/server/libjvm.so"; \
    fi && \
    if [ -f "$JH/lib/libjava.so" ]; then \
        ln -sf "$JH/lib/libjava.so" "$JH/jre/lib/amd64/libjava.so"; \
    fi

WORKDIR /app

ENV PYLUCENE_VERSION=9.12.0

RUN echo "=== Downloading PyLucene ${PYLUCENE_VERSION} ===" && \
    wget --progress=dot:giga --timeout=60 --tries=3 --no-check-certificate \
        https://downloads.apache.org/lucene/pylucene/pylucene-${PYLUCENE_VERSION}-src.tar.gz \
        -O pylucene-${PYLUCENE_VERSION}-src.tar.gz || \
    (echo "Primary mirror failed, trying archive mirror..." && \
     wget --progress=dot:giga --timeout=60 --tries=3 --no-check-certificate \
        https://archive.apache.org/dist/lucene/pylucene/pylucene-${PYLUCENE_VERSION}-src.tar.gz \
        -O pylucene-${PYLUCENE_VERSION}-src.tar.gz) && \
    echo "Download complete. File size:" && \
    ls -lh pylucene-${PYLUCENE_VERSION}-src.tar.gz

RUN echo "=== Extracting PyLucene ===" && \
    tar -xzf pylucene-${PYLUCENE_VERSION}-src.tar.gz && \
    echo "=== Building JCC (step 1/3) ===" && \
    cd pylucene-${PYLUCENE_VERSION}/jcc && \
    JCC_JDK=${JAVA_HOME} python3 setup.py build && \
    JCC_JDK=${JAVA_HOME} python3 setup.py install && \
    echo "JCC build complete"

RUN echo "=== Building PyLucene (step 2/3 - this will take 10-20 minutes) ===" && \
    cd /app/pylucene-${PYLUCENE_VERSION} && \
    make all JCC="python3 -m jcc" ANT=ant PYTHON=python3 NUM_FILES=8 && \
    echo "=== Installing PyLucene (step 3/3) ===" && \
    make install JCC="python3 -m jcc" ANT=ant PYTHON=python3 NUM_FILES=8 && \
    cd /app && \
    rm -rf pylucene-${PYLUCENE_VERSION} pylucene-${PYLUCENE_VERSION}-src.tar.gz && \
    echo "=== Verifying PyLucene installation ===" && \
    python3 -c "import lucene; print('✓ PyLucene installed successfully')" || \
    (echo "✗ ERROR: PyLucene installation verification failed!" && exit 1)

FROM python:3.10-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${JAVA_HOME}/lib:${JAVA_HOME}/jre/lib/amd64/server:${LD_LIBRARY_PATH}"

RUN JH="$JAVA_HOME" && \
    mkdir -p "$JH/jre/lib/amd64/server" && \
    if [ -f "$JH/lib/server/libjvm.so" ]; then \
        ln -sf "$JH/lib/server/libjvm.so" "$JH/jre/lib/amd64/server/libjvm.so"; \
    fi && \
    if [ -f "$JH/lib/libjava.so" ]; then \
        ln -sf "$JH/lib/libjava.so" "$JH/jre/lib/amd64/libjava.so"; \
    fi

RUN mkdir -p /usr/local/lib/python3.10/site-packages
COPY --from=pylucene-builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

WORKDIR /app

ENV PYTHONPATH=/app:${PYTHONPATH}

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

COPY . .
RUN chmod +x /app/scripts/docker-entrypoint.sh

RUN mkdir -p data/raw index/lucene_index

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import streamlit; print('OK')" || exit 1

ENTRYPOINT ["bash", "/app/scripts/docker-entrypoint.sh"]
