FROM python:3.10.5

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

WORKDIR /app

RUN pip install poetry \
    && poetry config virtualenvs.in-project false

# Mecab
RUN apt update \
  && apt install -y --no-install-recommends mecab libmecab-dev mecab-ipadic-utf8

# # Install CRF++
RUN wget -O /tmp/CRF++-0.58.tar.gz "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7QVR6VXJ5dWExSTQ" \
  && cd /tmp \
  && tar zxf CRF++-0.58.tar.gz \
  && cd CRF++-0.58 \
  && ./configure \
  && make \
  && make install \
  && cd / \
  && rm /tmp/CRF++-0.58.tar.gz \
  && rm -rf /tmp/CRF++-0.58 \
  && ldconfig

# Install CaboCha
RUN cd /tmp \
  && FILE_ID="0B4y35FiV1wh7SDd1Q1dUQkZQaUU" \
  && curl -c /tmp/cookie -s "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null \
  && CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  \
  && curl -b /tmp/cookie -L "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o cabocha-0.69.tar.bz2 \
  && tar jxf cabocha-0.69.tar.bz2 \
  && cd cabocha-0.69 \
  && ./configure --with-mecab-config=`which mecab-config` --with-charset=UTF8 \
  && make \
  && make install \
  && cd python \
  && python setup.py build \
  && python setup.py install \
  && cd / \
  && rm /tmp/cookie \
  && rm /tmp/cabocha-0.69.tar.bz2 \
  && rm -rf /tmp/cabocha-0.69 \
  && ldconfig

# Download lf-comma-period model weights
RUN FILE_ID="10-Gt9GqGgJCXQBitYxB6zez2Y4Mg9ZzK" \
  && curl -c /tmp/cookie -s "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null \
  && CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  \
  && curl -b /tmp/cookie -L "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o lf_comma_period_model.ckpt \
  && rm /tmp/cookie

COPY . .
RUN poetry install --no-root

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
