import streamlit as st
from PIL import Image

st.header("Что такое нейронная сеть ")
st.subheader("Понятие")
st.write("Искусственная нейронная сеть, она же нейронная сеть, она же нейросеть — это математическая модель, основанная на принципах нейронных сетей живых организмов, то есть имитирующая работу мозга.  Ее основное назначение — решать интеллектуальные задачи. Те, для которых невозможно задать универсальный алгоритм действий обычным программированием. ")
st.write("Концепция нейронных сетей была сформулирована в 1943 году Уолтером Питтсом и Уорреном МакКалоком, но только в 1958 была представлена первая простейшая модель однослойной сети — перцептрона. Однако развитие этой сферы было существенно ограниченно в то время — обучение модели требует больших вычислительных мощностей. ")
st.write("В наше время, с увеличением мощностей компьютеров, стало возможно обучение масштабных нейронных сетей. В настоящем нейросети у всех на слуху, благодаря таким моделям как MidJorney, Stable Diffusion, ChatGPT и другим. Хотя их возможности и поражают наше воображение, однако и в наше время нейросети могут сравниться по размерам разве что с мозгом мухи и не могут одновременно служить более чем для одной задачи.  ")
st.write("На нашем сайте можно найти инструкции по обучению своих нейросетей, классифицирующих наборы данных или изображений. А также информацию о том, как подготовить набор данных для старта. ")

st.subheader("Применение нейронных сетей ")
st.write("Нейросети могут применяться во множестве отраслей и решать не меньшее множество задач:")
st.markdown(" - Классификация (оно же распознавание) — заключается в распределении объектов по группам по их признакам; ")
st.markdown(" - Регрессия — отличается от классификации тем, что объекту присуждается какое-либо численное значение, а не группа; ")
st.markdown(" - Прогнозирование — заключается в попытках предсказать следующие значения числового ряда; ")
st.markdown(" - Кластеризация — похожа на классификацию, но итоговые классы не определены заранее, а определяются в процессе выполнения самой нейросетью;  ")
st.markdown(" - Генерация — заключается в имитации творческой деятельности. Эту задачу выполняют нейронные сети рисующие картины, пишущие произведения и сочиняющие музыку;")
st.markdown(" - Сжатие данных и ассоциативная память — заключается в нахождении нейросетью закономерностей во входных данных и ужимании объёма данных. А также способности восстановления всей информации по её части;")
st.info("Однако для каждой задачи применяется своя архитектура, и модель нейронной сеть должна быть заточена на её решение, а потому с, допустим, классифицирующим строением, нейронная сеть не сможет ничего генерировать.")

st.subheader("Устройство нейронных сетей ")
with Image.open("nn1.jpeg") as f:
    st.image(f)
    st.caption("Это пример архитектуры многослойной нейронной сети.")
st.write("Нейросети имеют различную внутреннюю структуру, которая зависит от их цели и задач. Однако все нейросети сходятся в одном: они состоят из множества нейронов, и связей между ними. Главная функция нейрона - обрабатывать входящие в него значения, хранить их и передавать. Нейроны объединяются в группы, которые называются слоями.  ")
st.write("Каждый слой (кроме первого и последнего) получает входные значения от всех нейронов предыдущего слоя и передает выходные значения всем нейронам следующего слоя. Первый слой — это входной слой, куда подаются данные, которые обрабатывает нейросеть. Последний слой — это выходной слой, куда по окончании передаётся результат работы нейросети. Связи между нейронами имеют определенные величины, которые называются весами. Именно настройка весов является ключевым этапом обучения нейросетей.")
st.write("Некоторые типы нейронных сетей могут содержать дополнительные элементы и связи, которые увеличивают их функциональность и эффективность. Однако базовая архитектура нейросети состоит из входного слоя, выходного слоя и одного или нескольких (а может не содержать) скрытых слоев между ними. Число и размер скрытых слоев определяет сложность задач, которые может решать нейросеть. Однако с ростом числа скрытых слоев также экспоненциально увеличивается время и трудность обучения нейросети. ")

st.subheader("Классификация нейронных сетей ")
st.write("Модели нейросетей, в зависимости от количества слоёв можно делить на: ")
st.markdown(" - Однослойные (состоят только из входного и выходного слоя);")
st.markdown(" - Многослойные (более сложные, кроме входного и выходного слоёв содержат некоторое число скрытых слоёв между ними).")
st.write("В зависимости от типа входных данных на: ")
st.markdown(" - Аналоговые (на вход подаются действительные числа); ")
st.markdown(" - Двоичные (на каждый вход подаются только 0 или 1); ")
st.markdown(" - Образные (на вход подаются нечисловая информация). ")
st.write("По направлению передачи значений на: ")
st.markdown(" - Прямого распространения (информация передаётся от входа к выходу, от предыдущего слоя к следующему, не возвращаясь на уже пройденные нейроны); ")
st.markdown(" - С функцией обратного распространения (рекуррентные) (значения могут передаваться на уже пройденные нейроны, как бы имитирую краткосрочную память). ")
st.write("По способу обучения на: ")
st.markdown(" - Использующие обучение с учителем (требуют заранее известных правильных ответов, на основе которых проверяются результаты); ")
st.markdown(" - Использующие обучение без учителя (не требуют ответов, обучаются сами);")
st.markdown(" - Смешанные (использующие методы как обучения с учителем, так и без него). ")
st.info("Опять же, для каждой отдельной задачи свой подход и свой тип нейросети. ")