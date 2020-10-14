import os

from setuptools import setup, find_packages

cur_dir = os.path.realpath(os.path.dirname(__file__))

if __name__ == "__main__":
    setup(
        name='MatBERT',
        version="0.0.1",
        python_requires='>=3.6',
        author="Haoyan Huo",
        author_email="haoyan.huo@lbl.gov",
        license="MIT License",
        packages=find_packages(),
        zip_safe=False,
        install_requires=open(os.path.join(cur_dir, 'requirements.txt')).readlines()
    )
