from setuptools import find_packages, setup

# Phương thức get_requirements(path: str) thực hiện loại bỏ các khoảng trắng 
# trong các đoạn văn bản được trích suất từ đường dẫn
def get_requirements(path: str):
    return [l.strip() for l in open(path)]

# Cài đặt mô hình llama 3
setup(
    name="llama3",
    version="0.0.1",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt"),
)
