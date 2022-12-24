import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='analysis',
    version='0.0.33',
    author='Nazir Nayal',
    author_email='nnayal17@ku.edu.tr',
    description='Tools for Deep Learning Related Data Analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/NazirNayal8/analyze',
    project_urls = {
        "Bug Tracker": "https://github.com/NazirNayal8/analyze/issues"
    },
    license='MIT',
    packages=['analysis'],
    install_requires=[
        
    ],
)