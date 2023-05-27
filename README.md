
# Интерактивная платформа для мониторинга внутренней отделки квартиры

Веб-приложение, которое позволяет на основе анализа видеопотока осуществлять мониторинг процессов внутренней отделки строящихся зданий.

Решение от команды "OTSO United".


## Deployment on docker

Для развертывания контейнера 

```bash
  docker build -t lct_otso_flask .
  docker run --gpus=all --restart=always --name lct_hack_otso -ti -p 5004:5005 lct_otso_flask
```

## Deployment on venv

Для развертывания приложения через виртуальное окружение необходимо установить python3.9

Создайте и активируйте виртуальное окружение 
```bash
  python3.9 -m venv venv
```
Windows 
```bash
  venv/Scripts/bin/activate.bat
```
Ubuntu 
```bash
  source venv/bin/activate
```
Установка зависимостей
```bash
  python3.9 -m pip install -r requirements.txt
```

## Authors

- [Илья Алиев](https://www.github.com/alievilya)
- [Андрей Безбородов](https://github.com/andreibezborodov)
- [Григорий Макаров]()


## Demo

![image demo](https://github.com/alievilya/lct_hackaton_web/blob/main/flask_app/data/git_demo/demo1.png?raw=true)

