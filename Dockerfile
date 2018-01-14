FROM tiagopeixoto/graph-tool

RUN pacman -S --noconfirm python-pip

RUN pacman -Sy

RUN pip install -r requirements.txt