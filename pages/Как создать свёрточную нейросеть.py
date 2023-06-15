import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import io

@st.cache_resource
def load_model():
    return EfficientNetB0(weights ='imagenet')


def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_image():
    uploaded_file = st.file_uploader(label='Загрузите фото зверюшки')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def print_predictions(preds):
    classes = decode_predictions(preds, top=10)[0]
    res = ''
    for cl in classes:
        if(cl[1]=="otter"):
            if cl[2] < .5 : res = "Возможно, это выдра"
            if cl[2] < .7 and cl[2] > .5: res = "Мне кажется, это выдра"
            if cl[2] > .7: res = "Это пределённо выдра!"
            break
        else: res = "Это не выдра. Загрузи картинку выдры!"
    st.write(res)

st.title('Выдра ли это?')
img = load_image()
result = st.button('Это выдра?')
model = load_model()
if result:
    st.write('Дай ка погляжу...')
    x = preprocess_image(img)
    preds = model.predict(x)
    print_predictions(preds)
if 1:
    st.header('Как построить и обучить полносвязную нейронную сеть на своём наборе данных')
    st.write('Свёрточная нейросеть в основном используется для работа с изображениями, она нацелена на эффективное распознавание образов на картинке. Если нужно понять, что за зверь изображён на фото, вы по адресу.')
    st.write('Для создания и обучения своей маленькой нейросети понадобится датасет, с объектами (классами, которые будет распознавать нейронная сеть')
    st.info("Код, представленный дальше, является шаблоном, вам необщодимо самим подставить необходимые значения при использовании. Для удобства можно открыть пример использования на примере выдр.")
    st.subheader('Подготовка датасета')
    st.write("Для начала переместите датасет на ваш аккаунт Google Диск и откройте Colaboratory на том же аккаунте. Создайте новый блокнот. Для того, чтобы создать что-нибудь простенькое нам понадобится вот такой обьём инструментов: ")
    st.info("Каждый блок кода нужно вставлять в отдельную ячейку в Google Colaboratory")
    st.code('''#Импорт необходивых библиотек 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from random import randint as rand
from google.colab import drive
import csv
import shutil
%matplotlib inline''')
    st.write("Ваш датасет нужно загрузить и распаковать. Для этого, в следующих блоках выполняем следующие команды: ")
    st.code('''# Импорт из Google Диска и распаковка набора данных
drive.mount("/content/gdrive", force_remount = True)
!unzip / content / gdrive / "MyDrive" / Имя архиваированного файла с датасетом*''')
    but1 = st.button("Пример с выдрами")
    if(but1):
        st.code('''# Импорт и распаковка набора данных с выдрами
        drive.mount("/content/gdrive", force_remount = True)
        !unzip / content / gdrive / "MyDrive" / "archive.zip"''')
    st.write("Далее внимательно посмотрите на внутреннее устройство вашего набора данных, а точнее как расположены фото для обучения: в одной папке или в нескольких, по классам объектов. В случае, если всё сложено в одну кучу, нам придётся сортировать по отдельным папкам, ну а если всё по полочкам, можете пропустить следующий пункт. ")
    st.write("В случае если у вас куча фото в одной папке, то где-то неподалёку обязательно должен лежать csv файл с распределением этой кучи по классам, запомнике как в этом файле называются столбцы. Для того, чтобы решить проблему нам потребуется написать простенькую функцию, сортирующую фото")
    st.code('''#В случае проблем с представлением данных в датасете
with open('Путь до csv файла*', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        where = "train/train/"+row[Название столбца с изображениями (файлами)*]
            shutil.move(where, "train/train/"+row[Название столбца с ответами (классами)*])''')
    but2 = st.button("Пример с выдрами", key = 1)
    if (but2):
        st.write("Так как в моём случае я хочу разделить все фото на выдр и нет, у меня получается следующая функция:")
        st.code('''#В случае проблем с представлением данных в датасете
        with open('train.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                where = "train/train/"+row['Image_id']
                if row['Animal'] == "otter":
                    shutil.move(where, "train/train/otter")
                else: 
                    shutil.move(where, "train/train/not_otter")''')
    st.write("Но вам нужно написать свою, конкретно под ваш пример. ")
    st.write("Теперь необходимо перевести наш датасет в датасет tensorflow. Для этого используем следующую функцию: ")
    st.code('''#Загрузка датасета в tensorflow
train = image_dataset_from_directory("train/train",
                                    seed=228,
                                    batch_size=256,
                                    image_size=(256, 256))
# Просмотр получившихся названий классов (какие объекты будет искать нейросеть)
class_names = train.class_names
class_names''')
    st.write("Где первым аргументом введите путь к папке с папками классов, а последним размер изображений.")
    st.write("Вы можете просмотреть какие данные загрузили, запустив следующий код: ")
    st.code('''#Печать короткой выборки изображений
plt.figure(figsize = (8, 8))
for images, labels in train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[1]])''')
    st.write("И наконец, пропитываем следующие команды для ускорения обучения нейросети в будущем.")
    st.code('''AUTOTUNE = tf.data.experimental.AUTOTUNE
train = train.prefetch(buffer_size=AUTOTUNE)''')
    st.write("Датасет подготовлен")
    st.subheader("Подготовка модели")
    st.write("Теперь настало время самой нейросети. Для этого создаём модель и последовательно добавляем ей скрытые слои, и в конце выходной слой (как на фото). Если вам интересно, что такое сверточный слой, слой подвыборки и как в принципе устроена нейросеть, то вам лучше загуглить это, так как тема довольна обширна. ")
    st.code('''# Описание модели
model = Sequential()
model.add(Conv2D(16, (5, 5), padding='same', 
                 input_shape=(256, 256, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
# Выходной слой, указываем здесь количество классов (варианотов ответов) (в моём случае 2)
model.add(Dense(2, activation='softmax'))

#После описания необходимо скомпилировать модель, превратив её в полноценный объект.
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])''')
    st.write("Модель готова")
    st.subheader("Обучение и проверка модели")
    st.write("Теперь самое важное, нашу модель нужно обучить на том датасете, который был загружен в tensorflow. Сделать это с помощью наших инструментов на удивление просто, нужно только запустить функцию: ")
    st.code('''#Обучение модели
history = model.fit(train, 
                epochs=5,
                verbose=2)''')
    st.caption('Это займёт какое-то время...')
    st.write("Поздравляю, если всё было сделано по рецепту и задача для нейросети де слишком сложная, то у вас получилась готовая модель нейросети, готовая различать объекты по фото")
    st.write("Если вы проделали всю эту работу не просто так и хотите испытать вашу модель, для этого скачайте из интернета какую-нибудь фотографию и переместите ее в главную папку colabratori. Так же составьте следующую функцию в отдельном блоке, где в качестве параметра введите адрес вашей фотографии")
    st.code('''# Загрузка изображения
img = image.load_img(Адрес изображения*, target_size=(256, 256))
# Печать  
plt.imshow(img)
plt.show()
# Преобразование
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# Прогон через сеть и выдача результатов
result = model.predict(x)
classes = decode_predictions(result, top=5)[0]
for cl in classes:
    print(cl[1], cl[2])''')
    st.write(
        "В случае, если у вас не получилось обучить нейросеть, внимательно проверьте содержимое вашего датасета, возможно фотографии распределены неправильно")
    st.write(
        "Также вашу нейросеть можно сохранить и загрузить для последующего использования, для этого достаточно выполнить следующий блок:")
    st.code('''#Сохранение модели
model.save("mywork.h5")
files.download("mywork.h5")''')
    st.caption('\n\n\n\n\nСделано командой “Меньше, чем три” (Команда 090302_4) ')
