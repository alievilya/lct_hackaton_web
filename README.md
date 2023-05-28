
# Интерактивная платформа для мониторинга внутренней отделки квартиры

Веб-приложение, позволяющее на основе анализа видеозаписи осуществлять мониторинг процессов внутренней отделки строящихся зданий.

После обработки видео на странице выводится таблица с подсчитанными процентами готовности каждого элемента отделки.

Решение от команды "OTSO United".

Макет с веб-интерфейсом доступен по адресу http://otsounitedlct.ru. Вычисления производятся на видеокарте GeForce GTX 1060.

## Mobile App
Также было реализовано мобильное приложение, выполняющее обработку в режиме реального времени. Решение находится в репозитории https://github.com/andreibezborodov/lct_hackaton_mobile

apk-файл доступен на диске https://drive.google.com/file/d/1ypbvAIwjVsarYj44d-bseCWpDPXbOjfj/view?usp=sharing

## Deployment web using Docker

Для развертывания контейнера 

```bash
  docker build -t lct_otso_flask .
  docker run --gpus=all --restart=always --name lct_hack_otso -ti -p 5004:5005 lct_otso_flask
```

## Deployment web using venv

Для развертывания приложения через виртуальное окружение необходимо установить python3.9

### Создайте и активируйте виртуальное окружение 
```bash
  python -m venv venv
```
Windows 
```bash
  venv/Scripts/bin/activate.bat
```
Ubuntu 
```bash
  source venv/bin/activate
```
### Установка зависимостей
```bash
  python -m pip install -r requirements.txt
```
### Запуск веб-приложения
```bash
  cd flask_app
  python flask_app.py
```

## Demo

Detection example

![Detection example](https://github.com/alievilya/lct_hackaton_web/blob/main/flask_app/data/git_demo/demo1.gif)



## Screenshots
![image demo](https://github.com/alievilya/lct_hackaton_web/blob/main/flask_app/data/git_demo/demo2.png?raw=true)

## Ссылка на презентацию с описанием проекта
https://docs.google.com/presentation/d/1pdLg4HP2bDWk_S4zBV3a4YkQaFsRRWI6eeVPBZkxnNY/edit?usp=sharing

## Authors

- [Илья Алиев](https://www.github.com/alievilya)
- [Андрей Безбородов](https://github.com/andreibezborodov)
- [Григорий Макаров](https://github.com/grifon-239)
- 