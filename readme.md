# ML Word Generator
Принимает в качестве входных данных один текстовый файл, где предполагается, что каждая строка является одним обучающим элементом, и генерирует больше подобных элементов. По сути, это авторегрессионная языковая модель на уровне символов с широким выбором моделей - от биграмм до трансформера (как в GPT). Например, мы можем предоставить ему базу данных имен, и он сгенерирует классные идеи детских имен, которые все звучат как имена, но не являются уже существующими именами.
# Легкий старт
Для легкого старта можете воспользоваться ***setup.bat***, а затем для демонстрации процесса обучения открыть ***test_learning.bat***, или же самостоятельно установить все зависимости.

Для дальнейшего использования обученных моделей в любых других проектах используйте ***mlwg.py***, импортируя функцию **gen**.

> **Важные замечания**
> 1. если вы хотите дообучить нейронную сеть до желаемого результата используйте аргумент --resume, иначе ваша предыдущая работа будет утеряна
> 2. в случае изменения параметров нейронной сети при обучении, перед использованием внесите правки в конфигурацию ***mlwg.py*** (класс **ModelConfig**)
> 3. не прерывать обучение до его автоматического завершения, иначе модель может сломаться