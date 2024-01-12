FROM python:slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wkhtmltopdf \
    xvfb \
    xauth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY . .

# Copy the NotoColorEmoji.ttf file into the container
COPY NotoColorEmoji.ttf /usr/share/fonts/truetype/NotoColorEmoji.ttf

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "get_emoji.py"]
