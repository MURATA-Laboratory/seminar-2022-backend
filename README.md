# seminar-2022-backend
## Overview
FastAPI server for subtitle generation.

## Prerequisites
Before you begin, ensure you have met the following requirements:

- You have installed `seminar-2022-backend/pyproject.toml`
- You have a `<Windows(wsl2 ubuntu)/Linux/Mac>`

## Installing poetry
To install poetry, follow these steps:
```
    curl -sSL https://install.python-poetry.org | python3 -
```


## Usage

1.Clone this repositry
```rb
    git clone git@github.com:MURATA-Laboratory/seminar-2022-backend.git
```

2.image build
```rb
    docker build -t poetryi -f Dockerfile .
```

3.container create
```rb
    docker create --name poetryc -p 127.0.0.1:8000:8000 poetryi
```
4.docker start 
```rb
    docker start poetryc
```
