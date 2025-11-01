# Демонстрация по предмету «статистическая физика»

## Настройка окружения
1. Установите Python 3.11 (проект разрабатывался под эту версию; локально можно использовать более новую, но PyInstaller должен совпадать с целевой версией Python).
2. Создайте и активируйте виртуальное окружение:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate            # macOS/Linux
   # или
   .\.venv\Scripts\activate.ps1         # Windows PowerShell
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Сборка приложения
PyInstaller использует файл `GradientTemperature.spec`, в котором уже перечислены конфигурация и необходимые ресурсы (`config.json`, `languages/`, `_internal/`).

### macOS (локальная сборка)
```bash
./scripts/build-mac.sh
```
Артефакт появится в `dist/GradientTemperature`. Сценарий ожидает активированное окружение `.venv` либо путь в переменной `PYINSTALLER_BIN`.

### Linux
Сборка должна выполняться в реальной Linux-среде, иначе бинарник может не запуститься из-за несовместимой glibc.
```bash
./scripts/build-linux.sh
```
Если запускаете скрипт из контейнера, примонтируйте проект и убедитесь, что установлены системные зависимости SDL, нужные `pygame`.

### Windows
```powershell
.\scripts\build-windows.ps1
```
Скрипт так же ожидает, что активирован виртуальное окружение (`.venv\Scripts\Activate.ps1`) с установленными зависимостями.

## Как собрать под все платформы с macOS
- PyInstaller не умеет кросс-компилировать, поэтому macOS не может напрямую собрать `.exe` или Linux ELF. Нужны среды с соответствующей ОС: виртуальные машины (Parallels, UTM/VirtualBox), Docker-контейнер для Linux или CI (GitHub Actions, GitLab CI, Azure Pipelines) с матрицей по платформам.
- Для Windows рекомендуемый вариант — GitHub Actions с `runs-on: windows-latest` либо облачный/локальный Windows-хост. Сценарий `scripts/build-windows.ps1` можно запускать прямо в workflow.
- Для Linux удобно использовать контейнер `pyinstaller/pyinstaller:latest` (или любой дистрибутив, поддерживающий нужную glibc). Запустите `./scripts/build-linux.sh` внутри контейнера.
- macOS бинарник собирается напрямую на Mac. Если запускаете PyInstaller в окружении с ограниченными правами, пропишите `export PYINSTALLER_CONFIG_DIR="$PWD/.pyinstaller"` перед сборкой, чтобы кэши создавались в рабочей директории.

## Быстрая проверка
После сборки запустите бинарник на целевой платформе и убедитесь, что доступны все экраны и загружаются ресурсы (`languages/*.json`, изображения из `_internal/images/`). При необходимости обновите файл `GradientTemperature.spec`, чтобы добавить новые ассеты.

## Теоретический материал
- Латех-исходники находятся в каталоге `docs/` (`theory_ru.tex`, `theory_en.tex`). Скомпилируйте их в PDF и сохраните под именами `theory_ru.pdf` и `theory_en.pdf`.
- Поместите полученные файлы в каталог `_internal/theory/` рядом с другими ресурсами.
- Экран теории рендерит страницы через библиотеку [`pymupdf`](https://pypi.org/project/pymupdf/). Убедитесь, что пакет установлен в активном окружении (`pip install pymupdf` или `pipenv install pymupdf`).
